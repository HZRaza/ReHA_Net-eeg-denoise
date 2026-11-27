"""metrics_full.py - every per-epoch metric we want logged, so nothing has to be re-run later.

Two metric families are returned, because the two papers define them differently:
  * Zhang et al. (EEGdenoiseNet benchmark):  cc, rrmse_t (RMS error / RMS clean), rrmse_f (periodogram PSD, eq 8)
  * ReHA-Net paper (Table 3):                psnr, snr, reha_rrmse_t (RMS error / std clean), reha_rrmse_s (z-scored),
                                             reha_rrmse_f (FFT magnitudes), reha_rrmse (mean of the three)
All values are means over segments. cc_sd / rrmse_t_sd are SDs across segments. If n_levels is given and the
segments are laid out as n_levels equal blocks (the val/test layout), per-SNR-level means are returned too.
"""
import numpy as np
from eeg_utils import cc as _zhang_cc, rrmse_spectral as _zhang_rrmse_f, rrmse_temporal as _zhang_rrmse_t


def full_metrics(pred, ref, fs=256, n_levels=None, eps=1e-12):
    p = np.asarray(pred, np.float64); p = p.reshape(len(p), -1)
    r = np.asarray(ref, np.float64); r = r.reshape(len(r), -1)
    err = r - p
    mse = np.mean(err ** 2, axis=1)
    sd = r.std(axis=1)

    cc, rrmse_t, rrmse_f = _zhang_cc(p, r), _zhang_rrmse_t(p, r), _zhang_rrmse_f(p, r, fs)

    psnr = 20 * np.log10(np.max(np.abs(r), axis=1) / (np.sqrt(mse) + eps))
    snr = 10 * np.log10(np.sum(r ** 2, axis=1) / (np.sum(err ** 2, axis=1) + eps))
    reha_t = np.sqrt(mse) / (sd + eps)
    zr = (r - r.mean(1, keepdims=True)) / (sd[:, None] + eps)
    zp = (p - p.mean(1, keepdims=True)) / (p.std(1)[:, None] + eps)
    reha_s = np.sqrt(np.mean((zr - zp) ** 2, axis=1))
    Fr, Fp = np.abs(np.fft.rfft(r, axis=1)), np.abs(np.fft.rfft(p, axis=1))
    reha_f = np.sqrt(np.mean((Fr - Fp) ** 2, axis=1)) / (np.sqrt(np.mean(Fr ** 2, axis=1)) + eps)

    seg = dict(mse=mse, cc=cc, rrmse_t=rrmse_t, rrmse_f=rrmse_f, psnr=psnr, snr=snr,
               reha_rrmse_t=reha_t, reha_rrmse_s=reha_s, reha_rrmse_f=reha_f,
               reha_rrmse=(reha_t + reha_s + reha_f) / 3)
    out = {k: float(np.mean(v)) for k, v in seg.items()}
    out["cc_sd"], out["rrmse_t_sd"] = float(np.std(cc)), float(np.std(rrmse_t))

    for k in ("cc", "rrmse_t", "psnr", "snr"):
        out[k + "_by_snr"] = []
    if n_levels and len(p) % n_levels == 0:
        blk = np.arange(len(p)) // (len(p) // n_levels)
        for k in ("cc", "rrmse_t", "psnr", "snr"):
            out[k + "_by_snr"] = [float(np.mean(seg[k][blk == i])) for i in range(n_levels)]
    return out
