from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import nibabel as nib
from nilearn import image, plotting
import mne
import io
import base64
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tempfile
import os
from typing import Tuple
import uvicorn
import pandas as pd

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def _write_bytes_to_temp(b: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(b)
    tmp.flush()
    tmp.close()
    return tmp.name

def _load_fmri_from_bytes(b: bytes) -> Tuple[nib.Nifti1Image, float]:
    path = _write_bytes_to_temp(b, suffix=".nii.gz")
    img = nib.load(path)
    zooms = img.header.get_zooms()
    tr = float(zooms[3]) if len(zooms) > 3 else 2.0
    return img, tr

def _load_eeg_from_bytes(b: bytes, filename: str = None, sfreq_hint: float = 256.0) -> Tuple[mne.io.Raw, float]:
    if filename and filename.lower().endswith(".edf"):
        path = _write_bytes_to_temp(b, suffix=".edf")
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        return raw, float(raw.info["sfreq"])
    s = b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else str(b)
    try:
        df = pd.read_csv(io.StringIO(s))
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to parse EEG CSV")
    if "time" in df.columns:
        times = np.asarray(df["time"].values, dtype=float)
        ch_cols = [c for c in df.columns if c != "time"]
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
    info = mne.create_info(ch_names=ch_cols, sfreq=sfreq, ch_types=["eeg"] * len(ch_cols))
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw, sfreq

def preprocess_fmri_simple(img, fwhm=6.0):
    try:
        smoothed = image.smooth_img(img, fwhm=fwhm)
        cleaned = image.clean_img(smoothed, standardize=True, detrend=True)
    except Exception:
        cleaned = img
    return cleaned

def preprocess_eeg_simple(raw, l_freq=1.0, h_freq=40.0):
    raw_clean = raw.copy()
    try:
        raw_clean.load_data()
        raw_clean.filter(l_freq, h_freq, fir_design="firwin", verbose=False)
        nchan = int(raw_clean.info.get("nchan", raw_clean.get_data().shape[0]))
        ncomp = min(max(1, nchan - 1), 15)
        if ncomp >= 1:
            ica = mne.preprocessing.ICA(n_components=ncomp, random_state=97, max_iter="auto")
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
        raise ValueError("fMRI expected 4D image")
    x, y, z, t = data.shape
    n_rois = max(1, int(n_rois))
    region_means = np.zeros((t, n_rois), dtype=np.float32)
    xsplits = np.linspace(0, x, n_rois + 1, dtype=int)
    for i in range(n_rois):
        xs = xsplits[i]; xe = xsplits[i+1]
        if xe <= xs:
            continue
        region = data[xs:xe, :, :, :]
        region_reshaped = region.reshape(-1, t)
        region_means[:, i] = region_reshaped.mean(axis=0)
    denom = region_means.std(axis=0) + 1e-9
    region_means = (region_means - region_means.mean(axis=0)) / denom
    return region_means

def eeg_band_powers(raw, sfreq, epoch_length_sec=2.0):
    data = raw.get_data()
    n_channels, n_times = data.shape
    epoch_samples = max(1, int(round(epoch_length_sec * sfreq)))
    n_epochs = max(1, n_times // epoch_samples)
    bands = {"delta": (1,4), "theta": (4,8), "alpha": (8,12), "beta": (12,30), "gamma": (30,45)}
    out = np.zeros((n_epochs, len(bands)), dtype=np.float32)
    band_names = list(bands.keys())
    for ei in range(n_epochs):
        s = ei*epoch_samples; e = min(s+epoch_samples, n_times)
        epoch = data[:, s:e]
        if epoch.shape[1] < 2:
            ps = np.zeros((n_channels,1)); freqs = np.array([0.0])
        else:
            ps = np.abs(np.fft.rfft(epoch, axis=1))**2
            freqs = np.fft.rfftfreq(epoch.shape[1], d=1.0/sfreq)
        for bi,(bname,(low,high)) in enumerate(bands.items()):
            idx = np.where((freqs>=low)&(freqs<=high))[0]
            val = 0.0 if idx.size==0 else float(ps[:, idx].mean())
            out[ei, bi] = val
    denom = out.std(axis=0)+1e-9
    out = (out - out.mean(axis=0))/denom
    return out, band_names

def align_features(fmri_feats, fmri_tr, eeg_feats, eeg_epoch_sec):
    T_fmri = int(fmri_feats.shape[0])
    T_eeg = int(eeg_feats.shape[0]) if eeg_feats is not None else 0
    ratio = float(fmri_tr)/float(eeg_epoch_sec)
    eeg_per_tr = []
    for ti in range(T_fmri):
        center_epoch = int(round(ti*ratio))
        idxs = np.arange(center_epoch-1, center_epoch+2, dtype=int)
        if T_eeg == 0:
            if eeg_feats is not None and eeg_feats.shape[1]>0:
                eeg_per_tr.append(np.zeros((eeg_feats.shape[1],), dtype=np.float32))
            else:
                eeg_per_tr.append(np.zeros((1,), dtype=np.float32))
        else:
            idxs = np.clip(idxs, 0, T_eeg-1)
            eeg_per_tr.append(eeg_feats[idxs].mean(axis=0))
    eeg_per_tr = np.vstack(eeg_per_tr)
    T = min(T_fmri, eeg_per_tr.shape[0])
    fused = np.hstack([fmri_feats[:T], eeg_per_tr[:T]])
    return fused

def build_symbolic_image(label, emotion_score=0.0, size=(256,256)):
    im = Image.new("RGB", size, (255,255,255))
    draw = ImageDraw.Draw(im)
    lbl = int(label)
    if lbl==0:
        draw.ellipse((60,60,196,196), outline="black", width=4); text="Face-like"
    elif lbl==1:
        draw.polygon([(30,200),(128,40),(226,200)], outline="black", width=4); text="Landscape"
    else:
        draw.rectangle((60,60,196,196), outline="black", width=4); text="Object"
    alpha = int(np.clip(80*(float(emotion_score)+1.0),0,255))
    overlay = Image.new("RGBA", size, (255,0,0,alpha))
    im = im.convert("RGBA"); im = Image.alpha_composite(im, overlay); im = im.convert("RGB")
    return im, text

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

@app.post("/api/process")
async def process(fmri: UploadFile = File(...), eeg: UploadFile = File(...), eeg_epoch_sec: float = 2.0, n_rois: int = 6):
    try:
        fmri_bytes = await fmri.read()
        eeg_bytes = await eeg.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read uploaded files")
    try:
        img, tr = _load_fmri_from_bytes(fmri_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load fMRI: {e}")
    try:
        raw, sfreq = _load_eeg_from_bytes(eeg_bytes, filename=eeg.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load EEG: {e}")
    img_pre = preprocess_fmri_simple(img)
    raw_pre = preprocess_eeg_simple(raw)
    fmri_feats = fmri_region_means(img_pre, n_rois=n_rois)
    eeg_feats, band_names = eeg_band_powers(raw_pre, sfreq, epoch_length_sec=eeg_epoch_sec)
    fused = align_features(fmri_feats, tr, eeg_feats, eeg_epoch_sec)
    try:
        tp = min(max(0, int(fused.shape[0]//2)), img_pre.shape[3]-1)
        sel = image.index_img(img_pre, tp)
        fig = plotting.plot_stat_map(sel, display_mode="ortho", colorbar=False)
        brain_png = fig_to_base64(fig)
    except Exception:
        brain_png = ""
    try:
        sf = float(raw_pre.info.get("sfreq", 256.0))
        start = 0.0
        duration = min(10.0, raw_pre.n_times/sf)
        start_samp = int(round(start*sf))
        stop_samp = int(round((start+duration)*sf))
        if stop_samp <= start_samp:
            stop_samp = min(start_samp+1, raw_pre.n_times)
        data = raw_pre.get_data()[:, start_samp:stop_samp]
        times = np.arange(data.shape[1])/sf + start
        fig2, ax = plt.subplots(figsize=(6,3))
        for i in range(min(6, data.shape[0])):
            ax.plot(times, data[i] + i*5)
        ax.set_xlabel("Time (s)")
        ax.set_title("EEG excerpt")
        eeg_png = fig_to_base64(fig2)
    except Exception:
        eeg_png = ""
    pred_label = int((np.arange(fused.shape[0])//10)[-1] if fused.shape[0]>0 else 0)
    emotion_proxy = float(np.tanh(eeg_feats[-1, band_names.index("theta")]) if eeg_feats.shape[0]>0 and "theta" in band_names else 0.0)
    recon_im, recon_text = build_symbolic_image(pred_label, emotion_proxy)
    buf = io.BytesIO(); recon_im.save(buf, format="PNG"); buf.seek(0)
    recon_b64 = base64.b64encode(buf.read()).decode("utf-8")
    response = {
        "fmri_shape": img.shape if hasattr(img, "shape") else None,
        "fmri_tr": tr,
        "eeg_sfreq": sfreq,
        "fmri_feats_shape": fmri_feats.shape,
        "eeg_feats_shape": eeg_feats.shape,
        "fused_shape": fused.shape,
        "brain_png_b64": brain_png,
        "eeg_png_b64": eeg_png,
        "recon_png_b64": recon_b64,
        "recon_label": recon_text
    }
    return response

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
