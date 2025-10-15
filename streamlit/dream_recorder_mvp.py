"""
Dream Recorder (fMRI + EEG Fusion) - Streamlit MVP
"""

import streamlit as st
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, plotting
import mne
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import io
import os
import tempfile
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    optim = None
    TORCH_AVAILABLE = False
    from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def _write_bytes_to_temp(b, suffix):
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(b)
    tmp.flush()
    tmp.close()
    return tmp.name


def generate_synthetic_fmri(path, shape=(40, 40, 24, 60), tr=2.0, random_state=0):
    rng = np.random.RandomState(random_state)
    x, y, z, t = shape
    data = np.zeros(shape, dtype=np.float32)
    centers = [
        (int(x * 0.2), int(y * 0.5), int(z * 0.5)),
        (int(x * 0.8), int(y * 0.3), int(z * 0.6)),
        (int(x * 0.5), int(y * 0.8), int(z * 0.3)),
    ]
    for ti in range(t):
        frame = rng.normal(scale=0.1, size=(x, y, z)).astype(np.float32)
        for i, c in enumerate(centers):
            denom1 = max(1.0, (t / (i + 2)))
            denom2 = max(1.0, (t / (i + 3)))
            denom3 = max(1.0, (t / (i + 1)))
            cx = int(c[0] + 2 * np.sin(2 * np.pi * (ti / denom1)))
            cy = int(c[1] + 2 * np.cos(2 * np.pi * (ti / denom2)))
            cz = int(c[2] + 1 * np.sin(2 * np.pi * (ti / denom3)))
            xx, yy, zz = np.ogrid[:x, :y, :z]
            g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) / (2.0 * (2.5 + i) ** 2))
            frame += (0.5 * (i + 1)) * g
        data[:, :, :, ti] = frame
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    img.header.set_zooms((1.0, 1.0, 1.0, float(tr)))
    nib.save(img, path)
    return path


def generate_synthetic_eeg_csv(path, n_channels=8, sfreq=256, duration_sec=120, random_state=0):
    rng = np.random.RandomState(random_state)
    times = np.arange(0, duration_sec, 1.0 / sfreq)
    n_times = len(times)
    ch_names = [f'EEG{c+1}' for c in range(n_channels)]
    data = np.zeros((n_times, n_channels), dtype=np.float32)
    for ch in range(n_channels):
        data[:, ch] = (
            0.3 * np.sin(2 * np.pi * 6 * times * (1 + (ch % 3) / 3.0))
            + 0.2 * np.sin(2 * np.pi * 12 * times * (1 + (ch % 4) / 4.0))
            + 0.1 * np.sin(2 * np.pi * 30 * times * (1 + (ch % 2)))
            + 0.05 * rng.randn(n_times)
        )
    df = pd.DataFrame(data, columns=ch_names)
    df.insert(0, 'time', times)
    df.to_csv(path, index=False)
    return path


def load_fmri(file_like):
    if hasattr(file_like, 'read'):
        b = file_like.read()
        path = _write_bytes_to_temp(b, suffix='.nii.gz')
        img = nib.load(path)
    else:
        img = nib.load(file_like)
    zooms = img.header.get_zooms()
    tr = float(zooms[3]) if len(zooms) > 3 else 2.0
    return img, tr


def preprocess_fmri_simple(img, fwhm=6.0):
    try:
        smoothed = image.smooth_img(img, fwhm=fwhm)
        cleaned = image.clean_img(smoothed, standardize=True, detrend=True)
    except Exception:
        cleaned = img
    return cleaned


def load_eeg(file_like, sfreq_hint=256.0):
    if hasattr(file_like, 'read'):
        name = getattr(file_like, 'name', None)
        content = file_like.read()
        if name and name.lower().endswith('.edf'):
            path = _write_bytes_to_temp(content, suffix='.edf')
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            return raw, float(raw.info.get('sfreq', sfreq_hint))
        s = content.decode('utf-8') if isinstance(content, (bytes, bytearray)) else content
        df = pd.read_csv(io.StringIO(s))
    else:
        path = file_like
        if isinstance(path, str) and path.lower().endswith('.edf'):
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            return raw, float(raw.info.get('sfreq', sfreq_hint))
        df = pd.read_csv(path)
    if 'time' in df.columns:
        times = np.asarray(df['time'].values, dtype=float)
        ch_cols = [c for c in df.columns if c != 'time']
        data = df[ch_cols].values.T.astype(np.float32)
        if times.size > 1:
            diffs = np.diff(times)
            median_dt = float(np.median(diffs)) if diffs.size > 0 else 1.0
            sfreq = 1.0 / median_dt
        else:
            sfreq = float(sfreq_hint)
    else:
        ch_cols = list(df.columns)
        data = df.values.T.astype(np.float32)
        sfreq = float(sfreq_hint)
    info = mne.create_info(ch_names=ch_cols, sfreq=sfreq, ch_types=['eeg'] * len(ch_cols))
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw, sfreq


def preprocess_eeg_simple(raw, l_freq=1.0, h_freq=40.0):
    raw_clean = raw.copy()
    try:
        raw_clean.load_data()
        raw_clean.filter(l_freq, h_freq, fir_design='firwin', verbose=False)
        nchan = int(raw_clean.info.get('nchan', raw_clean.get_data().shape[0]))
        ncomp = min(max(1, nchan - 1), 15)
        if ncomp >= 1:
            ica = mne.preprocessing.ICA(n_components=ncomp, random_state=97, max_iter='auto')
            ica.fit(raw_clean)
            try:
                ica.exclude = [0]
                ica.apply(raw_clean)
            except Exception:
                pass
    except Exception:
        pass
    return raw_clean


def fmri_region_means(img, n_rois=6):
    data = img.get_fdata()
    if data.ndim != 4:
        raise ValueError('fMRI expected 4D image')
    x, y, z, t = data.shape
    n_rois = max(1, int(n_rois))
    region_means = np.zeros((t, n_rois), dtype=np.float32)
    xsplits = np.linspace(0, x, n_rois + 1, dtype=int)
    for i in range(n_rois):
        xs = xsplits[i]
        xe = xsplits[i + 1]
        if xe <= xs:
            region_means[:, i] = 0.0
            continue
        region = data[xs:xe, :, :, :]
        region_reshaped = region.reshape(-1, t)
        region_means[:, i] = region_reshaped.mean(axis=0)
    denom = region_means.std(axis=0) + 1e-9
    region_means = (region_means - region_means.mean(axis=0)) / denom
    names = [f'roi_x{i}' for i in range(region_means.shape[1])]
    return region_means, names


def eeg_band_powers(raw, sfreq, epoch_length_sec=2.0):
    data = raw.get_data()
    n_channels, n_times = data.shape
    epoch_samples = max(1, int(round(epoch_length_sec * sfreq)))
    n_epochs = max(1, n_times // epoch_samples)
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
    out = np.zeros((n_epochs, len(bands)), dtype=np.float32)
    band_names = list(bands.keys())
    for ei in range(n_epochs):
        s = ei * epoch_samples
        e = min(s + epoch_samples, n_times)
        epoch = data[:, s:e]
        if epoch.shape[1] < 2:
            ps = np.zeros((n_channels, 1), dtype=float)
            freqs = np.array([0.0], dtype=float)
        else:
            ps = np.abs(np.fft.rfft(epoch, axis=1)) ** 2
            freqs = np.fft.rfftfreq(epoch.shape[1], d=1.0 / sfreq)
        for bi, (bname, (low, high)) in enumerate(bands.items()):
            idx = np.where((freqs >= low) & (freqs <= high))[0]
            if idx.size == 0:
                val = 0.0
            else:
                val = float(ps[:, idx].mean())
            out[ei, bi] = val
    denom = out.std(axis=0) + 1e-9
    out = (out - out.mean(axis=0)) / denom
    return out, band_names


def align_features(fmri_feats, fmri_tr, eeg_feats, eeg_epoch_sec):
    T_fmri = int(fmri_feats.shape[0])
    T_eeg = int(eeg_feats.shape[0]) if eeg_feats is not None else 0
    ratio = float(fmri_tr) / float(eeg_epoch_sec)
    eeg_per_tr = []
    for ti in range(T_fmri):
        center_epoch = int(round(ti * ratio))
        idxs = np.arange(center_epoch - 1, center_epoch + 2, dtype=int)
        if T_eeg == 0:
            if eeg_feats is not None and eeg_feats.shape[1] > 0:
                eeg_per_tr.append(np.zeros((eeg_feats.shape[1],), dtype=np.float32))
            else:
                eeg_per_tr.append(np.zeros((1,), dtype=np.float32))
        else:
            idxs = np.clip(idxs, 0, T_eeg - 1)
            eeg_per_tr.append(eeg_feats[idxs].mean(axis=0))
    eeg_per_tr = np.vstack(eeg_per_tr)
    T = min(T_fmri, eeg_per_tr.shape[0])
    fused = np.hstack([fmri_feats[:T], eeg_per_tr[:T]])
    return fused


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, n_classes=3):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def create_sequences(X, y, seq_len=8):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError('X must be 2D')
    if len(X) <= seq_len:
        return np.zeros((0, seq_len, X.shape[1]), dtype=X.dtype), np.zeros((0,), dtype=int)
    Xs = []
    ys = []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(int(y[i + seq_len]))
    return np.stack(Xs), np.array(ys, dtype=int)


def _fig_to_image_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def plot_fmri_timepoint(img, timepoint=0):
    try:
        tp = int(timepoint)
        tp = max(0, min(tp, img.shape[3] - 1))
        sel = image.index_img(img, tp)
        fig = plt.figure(figsize=(6, 6))
        display = plotting.plot_stat_map(sel, display_mode='ortho', colorbar=True)
        plt.tight_layout()
        fig = plt.gcf()
        plt.close()
        return fig
    except Exception:
        try:
            tp = int(timepoint)
            tp = max(0, min(tp, img.shape[3] - 1))
            sel = image.index_img(img, tp)
            data = sel.get_fdata()
            midz = data.shape[2] // 2
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(data[:, :, midz].T, origin='lower')
            ax.set_title(f'fMRI timepoint {tp} (slice)')
            plt.close()
            return fig
        except Exception:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'Unable to render fMRI', ha='center')
            plt.close()
            return fig


def plot_eeg_raw(raw, start=0.0, duration=10.0):
    sfreq = float(raw.info.get('sfreq', 256.0))
    start_samp = int(max(0, int(round(start * sfreq))))
    stop_samp = int(min(int(getattr(raw, 'n_times', raw.get_data().shape[1])), int(round((start + duration) * sfreq))))
    if stop_samp <= start_samp:
        stop_samp = min(start_samp + 1, int(getattr(raw, 'n_times', raw.get_data().shape[1])))
    data = raw.get_data()[:, start_samp:stop_samp]
    times = np.arange(data.shape[1]) / sfreq + start
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(min(6, data.shape[0])):
        ax.plot(times, data[i] + i * 5, label=raw.ch_names[i])
    ax.set_xlabel('Time (s)')
    ax.set_title('EEG excerpt (stacked channels)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.close()
    return fig


def build_symbolic_image(label, emotion_score=0.0, size=(256, 256)):
    im = Image.new('RGB', size, (255, 255, 255))
    draw = ImageDraw.Draw(im)
    lbl = int(label)
    if lbl == 0:
        draw.ellipse((60, 60, 196, 196), outline='black', width=4)
        text = 'Face-like'
    elif lbl == 1:
        draw.polygon([(30, 200), (128, 40), (226, 200)], outline='black', width=4)
        text = 'Landscape'
    else:
        draw.rectangle((60, 60, 196, 196), outline='black', width=4)
        text = 'Object'
    alpha = int(np.clip(80 * (float(emotion_score) + 1.0), 0, 255))
    overlay = Image.new('RGBA', size, (255, 0, 0, alpha))
    im = im.convert('RGBA')
    im = Image.alpha_composite(im, overlay)
    im = im.convert('RGB')
    return im, text


st.set_page_config(page_title='Dream Recorder (fMRI + EEG Fusion) - MVP', layout='wide')
st.title('✨ Dream Recorder (fMRI + EEG Fusion) — MVP')

with st.sidebar:
    st.header('Controls')
    st.markdown('Upload real data (nii / nii.gz and csv/edf) or generate synthetic data for demo.')
    uploaded_fmri = st.file_uploader('Upload fMRI (.nii/.nii.gz)', type=['nii', 'nii.gz'])
    uploaded_eeg = st.file_uploader('Upload EEG (.csv/.edf)', type=['csv', 'edf'])
    if st.button('Generate synthetic demo data'):
        tmpdir = tempfile.mkdtemp()
        fmri_path = os.path.join(tmpdir, 'synthetic_fmri.nii.gz')
        eeg_path = os.path.join(tmpdir, 'synthetic_eeg.csv')
        generate_synthetic_fmri(fmri_path)
        generate_synthetic_eeg_csv(eeg_path, duration_sec=120)
        st.success(f'Synthetic data written to {tmpdir}')
        uploaded_fmri = open(fmri_path, 'rb')
        uploaded_eeg = open(eeg_path, 'rb')
    st.markdown('---')
    eeg_epoch_sec = st.number_input('EEG epoch length (s) for feature extraction', value=2.0, min_value=0.5)
    fmri_smoothing = st.number_input('fMRI smoothing fwhm (mm)', value=6.0)
    run_train = st.button('Train demo model')

col1, col2 = st.columns([1, 1])

img = None
raw = None
sfreq = None
tr = None
if uploaded_fmri is not None:
    try:
        img, tr = load_fmri(uploaded_fmri)
        st.success(f'Loaded fMRI (TR={tr:.2f}s)')
    except Exception as e:
        st.error(f'Failed to load fMRI: {e}')

if uploaded_eeg is not None:
    try:
        raw, sfreq = load_eeg(uploaded_eeg)
        st.success(f'Loaded EEG (sfreq={sfreq:.1f} Hz, channels={len(raw.ch_names)})')
    except Exception as e:
        st.error(f'Failed to load EEG: {e}')

if img is not None:
    st.header('fMRI preview')
    max_tp = int(img.shape[3] - 1) if getattr(img, 'ndim', 4) == 4 else 0
    tp = st.slider('Timepoint to visualize', 0, max_tp, 0)
    fig = plot_fmri_timepoint(img, timepoint=tp)
    st.pyplot(fig)

if raw is not None:
    st.header('EEG preview')
    eeg_start = st.number_input('EEG preview start (s)', value=0.0)
    eeg_dur = st.number_input('EEG preview duration (s)', value=10.0)
    fig2 = plot_eeg_raw(raw, start=eeg_start, duration=eeg_dur)
    st.pyplot(fig2)

if st.button('Preprocess & Extract Features'):
    if img is None or raw is None:
        st.error('Please provide both fMRI and EEG (or generate synthetic data).')
    else:
        with st.spinner('Preprocessing fMRI...'):
            img_pre = preprocess_fmri_simple(img, fwhm=fmri_smoothing)
        with st.spinner('Preprocessing EEG...'):
            raw_pre = preprocess_eeg_simple(raw)
        st.success('Preprocessing done')
        with st.spinner('Extracting fMRI region means...'):
            fmri_feats, roi_names = fmri_region_means(img_pre, n_rois=6)
        with st.spinner('Extracting EEG band powers...'):
            eeg_feats, band_names = eeg_band_powers(raw_pre, sfreq if sfreq is not None else 256, epoch_length_sec=eeg_epoch_sec)
        st.write('fMRI feat shape:', fmri_feats.shape)
        st.write('EEG feat shape:', eeg_feats.shape)
        fused = align_features(fmri_feats, tr, eeg_feats, eeg_epoch_sec)
        st.write('Fused features shape:', fused.shape)
        st.session_state['fmri_feats'] = fmri_feats
        st.session_state['eeg_feats'] = eeg_feats
        st.session_state['fused'] = fused
        st.session_state['roi_names'] = roi_names
        st.session_state['band_names'] = band_names

if 'fused' in st.session_state:
    fused = st.session_state['fused']
    st.header('Dataset & Model (demo)')
    T, F = fused.shape
    labels = np.array([(i // 10) % 3 for i in range(T)])
    st.write('Example labels (first 30):', labels[:30])
    if run_train:
        st.info('Training model on fused features (demo labels) ...')
        X = fused
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        seq_len = 8
        X_seq, y_seq = create_sequences(Xs, labels, seq_len=seq_len)
        if X_seq.shape[0] == 0:
            st.error('Not enough data to create training sequences. Reduce seq_len or provide more data.')
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = SimpleLSTM(input_size=F, hidden_size=64, num_layers=1, n_classes=3).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            n_epochs = 8
            batch_size = 16
            for epoch in range(n_epochs):
                model.train()
                perm = np.random.permutation(len(X_train))
                losses = []
                for i in range(0, len(perm), batch_size):
                    idx = perm[i:i + batch_size]
                    xb = torch.tensor(X_train[idx], dtype=torch.float32).to(device)
                    yb = torch.tensor(y_train[idx], dtype=torch.long).to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                model.eval()
                with torch.no_grad():
                    xv = torch.tensor(X_val, dtype=torch.float32).to(device)
                    yv = torch.tensor(y_val, dtype=torch.long).to(device)
                    pv = model(xv).cpu().numpy()
                    acc = accuracy_score(y_val, pv.argmax(axis=1))
                st.write(f'Epoch {epoch+1}/{n_epochs} — train_loss={np.mean(losses):.4f} val_acc={acc:.3f}')
            model = model.cpu()
            st.success('Training finished')
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['seq_len'] = seq_len

if 'model' in st.session_state and 'fused' in st.session_state:
    st.header('Inference & Reconstruction')
    model = st.session_state['model']
    scaler = st.session_state['scaler']
    seq_len = st.session_state['seq_len']
    fused = st.session_state['fused']
    Xs = scaler.transform(fused)
    if Xs.shape[0] < seq_len:
        st.error('Not enough timepoints for inference sequence.')
    else:
        seq = Xs[-seq_len:][None, ...]
        model.eval()
        with torch.no_grad():
            inp = torch.tensor(seq, dtype=torch.float32)
            preds = model(inp).cpu().numpy()[0]
            pred_label = int(preds.argmax())
            pred_scores = preds
        st.write('Predicted label:', pred_label, 'scores:', pred_scores)
        if 'band_names' in st.session_state and 'eeg_feats' in st.session_state:
            band_names = st.session_state['band_names']
            try:
                theta_idx = band_names.index('theta')
                emotion_proxy = st.session_state['eeg_feats'][-1, theta_idx]
                emotion_score = float(np.tanh(emotion_proxy))
            except Exception:
                emotion_score = 0.0
        else:
            emotion_score = 0.0
        recon_im, recon_text = build_symbolic_image(pred_label, emotion_score=emotion_score)
        st.image(recon_im, caption=f'Reconstructed symbolic image — {recon_text} (emotion proxy {emotion_score:.2f})')

if 'fused' in st.session_state:
    if st.button('Download fused features (.npz)'):
        fused = st.session_state['fused']
        bio = io.BytesIO()
        np.savez(bio, fused=fused)
        bio.seek(0)
        st.download_button('Download', data=bio.read(), file_name='dream_fused_features.npz')

st.markdown('---')
st.subheader('Background & Next steps')
st.markdown(
    """
    This MVP shows an end-to-end flow for combining fMRI (where) and EEG (when) into a fused feature
    representation, training a small temporal model, and producing a symbolic visualization.

    Next improvements for research / production:
    * Run a real fMRIPrep pipeline and use atlas-based ROI extraction (e.g., Schaefer, AAL, or Glasser).
    * Use event-related EEG analysis and better artifact detection (e.g., EOG/ECG automatic labeling).
    * Replace symbolic reconstruction with CLIP-guided VDVAE / Stable Diffusion image synthesis conditioned on
      predicted latent vectors (requires GPUs and careful safety review).
    * Improve model architecture with cross-attention multimodal transformers.

    """
)

st.markdown('**Caveats:** This demo is educational; please do not use it for clinical or diagnostic decisions.')
