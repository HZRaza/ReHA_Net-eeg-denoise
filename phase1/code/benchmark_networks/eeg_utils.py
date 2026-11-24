"""Numpy-only helpers shared by training, evaluation and the plot scripts.

No TensorFlow needed, so everything here can be tested and reused anywhere.

Metric families (both are computed; use the one that matches what you compare against)
  Zhang et al. 2021 (EEGdenoiseNet)  : cc, rrmse_temporal (/RMS of clean), rrmse_spectral (PSD based)
  ReHA-Net, Francis & Vadivu 2026    : psnr, snr, rrmse_t, rrmse_f, rrmse_s, rrmse_total
      PSNR = 20*log10(max|x| / sqrt(MSE))      SNR = 10*log10(sum x^2 / sum (x-xhat)^2)
      RRMSE_t = RMS(x-xhat)/std(x)             RRMSE_f = same on |FFT| magnitudes
      RRMSE_s = same on z-scored signals       RRMSE_total = (t + f + s)/3  <- the paper's 'RRMSE'
  (The paper's Table 5 confirms the last line: (0.1714+0.1785+0.1461)/3 = 0.1653.)

Also here: get_lr (learning-rate schedules), EarlyStopper, epoch_permutation, config_fingerprint.

Command line (re-computes every number from the saved test outputs):
  python eeg_utils.py path/to/nn_output [--fs 256] [--snr-low -7 --snr-high 2]
"""
import argparse
import hashlib
import json
import math
import os

import numpy as np

EPS = 1e-8


def _f64(x):
    return np.asarray(x, dtype=np.float64)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def rms(x, axis=-1):
    return np.sqrt(np.mean(np.square(_f64(x)), axis=axis))


def cc(pred, ref):
    """Pearson correlation per segment. Inputs (N, T) -> (N,)."""
    p, r = _f64(pred), _f64(ref)
    p = p - p.mean(axis=-1, keepdims=True)
    r = r - r.mean(axis=-1, keepdims=True)
    return (p * r).sum(-1) / np.sqrt((p ** 2).sum(-1) * (r ** 2).sum(-1))


def rrmse_temporal(pred, ref):
    """Zhang eq 7: RMS(pred - ref) / RMS(ref)."""
    return rms(_f64(pred) - _f64(ref)) / rms(ref)


def psd(x, fs, fmax=120.0):
    """Periodogram PSD, FFT length = segment length, 0..fmax Hz (Zhang et al.)."""
    x = _f64(x)
    n = x.shape[-1]
    p = np.abs(np.fft.rfft(x, n=n, axis=-1)) ** 2 / (fs * n)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    return p[..., f <= fmax]


def rrmse_spectral(pred, ref, fs):
    """Zhang eq 8: RMS(PSD(pred) - PSD(ref)) / RMS(PSD(ref))."""
    pp, pr = psd(pred, fs), psd(ref, fs)
    return rms(pp - pr) / rms(pr)


def psnr(pred, ref):
    """ReHA-Net eq 6: 20*log10(max|x| / sqrt(MSE))."""
    p, r = _f64(pred), _f64(ref)
    mse = np.maximum(np.mean((p - r) ** 2, axis=-1), 1e-20)
    return 20.0 * np.log10(np.max(np.abs(r), axis=-1) / np.sqrt(mse))


def snr(pred, ref):
    """ReHA-Net SNR: 10*log10(sum x^2 / sum((x-xhat)^2 + eps)) (output SNR, in dB)."""
    p, r = _f64(pred), _f64(ref)
    return 10.0 * np.log10((r ** 2).sum(-1) / (((r - p) ** 2) + EPS).sum(-1))


def rrmse_t(pred, ref):
    """ReHA-Net RRMSE (temporal): RMS(x - xhat) / (std(x) + eps)."""
    p, r = _f64(pred), _f64(ref)
    return rms(p - r) / (np.std(r, axis=-1) + EPS)


def rrmse_f(pred, ref):
    """ReHA-Net RRMSE (frequency): relative error of the FFT magnitudes."""
    fp = np.abs(np.fft.rfft(_f64(pred), axis=-1))
    fr = np.abs(np.fft.rfft(_f64(ref), axis=-1))
    return rms(fp - fr) / (rms(fr) + EPS)


def _z(x):
    x = _f64(x)
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + EPS)


def rrmse_s(pred, ref):
    """ReHA-Net RRMSE (scaled): error between z-scored signals, so amplitude is ignored."""
    zp, zr = _z(pred), _z(ref)
    return rms(zp - zr) / (np.std(zr, axis=-1) + EPS)


def segment_metrics(pred, ref, fs=256):
    """Every metric per segment: dict name -> (N,) array."""
    p = _f64(pred).reshape(len(ref), -1)
    r = _f64(ref).reshape(len(ref), -1)
    t, f, s = rrmse_t(p, r), rrmse_f(p, r), rrmse_s(p, r)
    return {
        "cc": cc(p, r), "psnr": psnr(p, r), "snr": snr(p, r),
        "rrmse_t": t, "rrmse_f": f, "rrmse_s": s, "rrmse_total": (t + f + s) / 3.0,
        "rrmse_temporal": rrmse_temporal(p, r), "rrmse_spectral": rrmse_spectral(p, r, fs),
        "mse": np.mean((p - r) ** 2, axis=-1),
    }


def metric_means(pred, ref, fs=256):
    """Mean of every metric over segments (used for the per-epoch validation log)."""
    return {k: float(np.mean(v)) for k, v in segment_metrics(pred, ref, fs).items()}


def measured_snr_db(clean, noisy, convention="standard"):
    """Actual input SNR of noisy = clean + noise, per segment.

    'standard' : 20*log10(RMS(clean)/RMS(noise))   (= 10*log10 of the power ratio)
    'repo'     : 10*log10(RMS(clean)/RMS(noise))   (Zhang et al. eq 2 / EEGdenoiseNet code)
    """
    clean = _f64(clean)
    noise = _f64(noisy) - clean
    k = 20.0 if convention == "standard" else 10.0
    return k * np.log10(rms(clean) / rms(noise))


_SUMMARY_KEYS = ("cc", "psnr", "snr", "rrmse_t", "rrmse_f", "rrmse_s", "rrmse_total",
                 "rrmse_temporal", "rrmse_spectral", "mse")


def summarize(denoised, clean, noisy=None, fs=256, n_levels=10, snr_levels=None):
    """Overall and per-SNR-block metrics (mean and SD over segments).

    The test/validation sets are n_levels equal blocks, one block per SNR level
    (lowest SNR first), each block holding the same segments.
    """
    n = len(clean)
    den = _f64(denoised).reshape(n, -1)
    cln = _f64(clean).reshape(n, -1)
    seg = segment_metrics(den, cln, fs)
    seg_noisy = None if noisy is None else segment_metrics(_f64(noisy).reshape(n, -1), cln, fs)

    def block(sl):
        r = {}
        for k in _SUMMARY_KEYS:
            r[k] = float(np.mean(seg[k][sl]))
            r[k + "_sd"] = float(np.std(seg[k][sl]))
        if seg_noisy is not None:
            for k in ("cc", "rrmse_temporal", "rrmse_t", "rrmse_total", "psnr", "snr"):
                r[k + "_noisy"] = float(np.mean(seg_noisy[k][sl]))
        return r

    out = {"n_segments": int(n), "overall": block(slice(None))}
    if n_levels > 1 and n % n_levels == 0:
        m = n // n_levels
        per = []
        for i in range(n_levels):
            r = block(slice(i * m, (i + 1) * m))
            r["snr_nominal_db"] = None if snr_levels is None else float(snr_levels[i])
            if noisy is not None:
                r["snr_input_db"] = float(np.mean(measured_snr_db(cln[i * m:(i + 1) * m],
                                                                  _f64(noisy).reshape(n, -1)[i * m:(i + 1) * m], "standard")))
            per.append(r)
        out["per_snr"] = per
    return out


def format_summary(s, snr_convention="repo"):
    o = s["overall"]
    lines = [f"Test segments: {s['n_segments']}",
             "  Zhang et al. metrics (EEGdenoiseNet)",
             f"    CC              : {o['cc']:.4f} +- {o['cc_sd']:.4f}" + (f"   (noisy input {o['cc_noisy']:.4f})" if "cc_noisy" in o else ""),
             f"    RRMSE temporal  : {o['rrmse_temporal']:.4f}" + (f"   (noisy input {o['rrmse_temporal_noisy']:.4f})" if "rrmse_temporal_noisy" in o else ""),
             f"    RRMSE spectral  : {o['rrmse_spectral']:.4f}   (PSD based)",
             f"    MSE             : {o['mse']:.5f}",
             "  ReHA-Net paper metrics (compare with its Tables 4/5)",
             f"    PSNR / SNR (dB) : {o['psnr']:.2f} / {o['snr']:.2f}",
             f"    RRMSE total     : {o['rrmse_total']:.4f}   (= mean of t, f, s)",
             f"    RRMSE t / f / s : {o['rrmse_t']:.4f} / {o['rrmse_f']:.4f} / {o['rrmse_s']:.4f}"]
    if "per_snr" in s:
        lines.append("")
        lines.append(f"  {'nominal':>8} {'actual':>8}   CC     RRMSEt(Zhang)  RRMSEtotal  PSNR   SNR")
        for r in s["per_snr"]:
            nom = r["snr_nominal_db"]
            if nom is None:
                lab = f"  {'?':>8} {'?':>8}"
            else:
                act = nom * 2 if snr_convention == "repo" else nom
                lab = f"  {nom:7.1f}dB {act:7.1f}dB"
            lines.append(f"{lab}  {r['cc']:.3f}  {r['rrmse_temporal']:.3f}          {r['rrmse_total']:.3f}      {r['psnr']:.1f}  {r['snr']:.1f}")
        if snr_convention == "repo":
            lines.append("  ('actual' = standard 20*log10 definition; the repo's eq.2 convention gives 2x the nominal dB)")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# learning-rate schedules
# ----------------------------------------------------------------------------
LR_SCHEDULES = ("constant", "step", "exp", "cosine", "cosine_restarts")


def get_lr(schedule, epoch, epochs, base_lr, min_ratio=0.01):
    """Learning rate for the 0-based `epoch`.

    constant         base_lr
    step             halves every 10 epochs
    exp              smooth decay to 10% of base_lr at the last epoch
    cosine           cosine decay to min_ratio*base_lr at the last epoch
    cosine_restarts  cosine annealing with warm restarts, T0 = 10, Tmult = 2 (as in ReHA-Net's Table 2)
    """
    if schedule == "constant":
        return float(base_lr)
    if schedule == "step":
        return float(base_lr * 0.5 ** (epoch // 10))
    if schedule == "exp":
        return float(base_lr * 0.1 ** (epoch / max(epochs - 1, 1)))
    if schedule == "cosine":
        t = epoch / max(epochs - 1, 1)
        return float(base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))))
    if schedule == "cosine_restarts":
        t_i, t_cur = 10, epoch
        while t_cur >= t_i:
            t_cur -= t_i
            t_i *= 2
        return float(base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t_cur / t_i))))
    raise ValueError("lr schedule must be one of %s, got %r" % (LR_SCHEDULES, schedule))


# ----------------------------------------------------------------------------
# training helpers
# ----------------------------------------------------------------------------
class EarlyStopper:
    """Tracks the best validation value and how long it has not improved.

    patience <= 0 disables stopping but the best epoch is still tracked.
    Epochs are 1-based.
    """

    def __init__(self, patience=10, min_delta=0.0, best=float("inf"), best_epoch=0, bad_epochs=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = best
        self.best_epoch = best_epoch
        self.bad_epochs = bad_epochs

    def update(self, value, epoch):
        """Returns (improved, should_stop)."""
        value = float(value)
        if value < self.best - self.min_delta:
            self.best, self.best_epoch, self.bad_epochs = value, int(epoch), 0
            return True, False
        self.bad_epochs += 1
        return False, bool(self.patience > 0 and self.bad_epochs >= self.patience)

    def state(self):
        return {"best": self.best, "best_epoch": self.best_epoch, "bad_epochs": self.bad_epochs}

    @classmethod
    def from_state(cls, state, patience, min_delta):
        return cls(patience, min_delta, state["best"], state["best_epoch"], state["bad_epochs"])


def epoch_permutation(seed, epoch, n):
    """Shuffle order for one epoch, reproducible from (seed, epoch) so resuming is exact."""
    return np.random.default_rng([int(seed), int(epoch)]).permutation(n)


def config_fingerprint(cfg):
    """Stable hash of the settings that must not change between a run and its resume."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------
# command line: recompute metrics from saved nn_output files
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nn_output", help="folder holding noiseinput_test.npy, Denoiseoutput_test.npy, EEG_test.npy")
    ap.add_argument("--fs", type=float, default=256, help="sampling rate: 256 for EOG, 512 for EMG")
    ap.add_argument("--snr-low", type=float, default=-7.0)
    ap.add_argument("--snr-high", type=float, default=2.0)
    ap.add_argument("--n-levels", type=int, default=10)
    ap.add_argument("--snr-convention", choices=["repo", "standard"], default="repo")
    ap.add_argument("--no-write", action="store_true", help="do not write metrics_summary.json")
    a = ap.parse_args()

    d = a.nn_output
    noisy = np.load(os.path.join(d, "noiseinput_test.npy"))
    den = np.load(os.path.join(d, "Denoiseoutput_test.npy"))
    clean = np.load(os.path.join(d, "EEG_test.npy"))
    levels = np.linspace(a.snr_low, a.snr_high, a.n_levels)
    s = summarize(den, clean, noisy, fs=a.fs, n_levels=a.n_levels, snr_levels=levels)
    print(format_summary(s, a.snr_convention))
    if not a.no_write:
        with open(os.path.join(d, "metrics_summary.json"), "w") as f:
            json.dump(s, f, indent=2)
        print("\nwrote", os.path.join(d, "metrics_summary.json"))


if __name__ == "__main__":
    main()
