"""Make every analysis plot for one finished run (or compare several runs).

One run:
  python plot_results.py "D:/.../results/EOG_simpleCNN_baseline/1"
  -> <run>/plots/*.png and three csv tables

Compare runs (efficiency study, ablations, baseline vs improved ...):
  python plot_results.py --compare run_a/1 run_b/1 run_c/1 --labels "lr 5e-5" "lr 1e-4" "cosine" --out compare_lr

The run folder is  results/<foldername>/1  (it holds nn_output/ and checkpoint/).
Needs numpy + matplotlib; scipy is optional (only for the classical high-pass baseline, plot 18).
Old runs (made before the metrics were logged each epoch) still work: plots that need
per-epoch PSNR / SNR / RRMSE variants are skipped and listed at the end.
"""
import argparse
import csv
import json
import os
import pickle
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eeg_utils import cc, psd, rms, segment_metrics, measured_snr_db  # noqa: E402

# ------------------------------------------------------------------ style (validated reference palette, light)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]  # fixed order, never cycled
NOISY_C, CLEAN_C, MODEL_C = MUTED, INK, BLUE
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.8, "lines.markersize": 4.5,
    "legend.frameon": False, "legend.fontsize": 9, "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"], "figure.dpi": 100, "savefig.dpi": 200,
})

PAPER_T5 = {"psnr": 27.10, "snr": 17.06, "cc": 0.976, "rrmse_total": 0.1653,
            "rrmse_s": 0.1714, "rrmse_t": 0.1785, "rrmse_f": 0.1461}
# ReHA-Net paper Table 4, EOG task (RRMSE, CC, SNR dB); None = not reported
PAPER_T4 = [("ReHA-Net", 0.165, 0.976, 17.06), ("MSCGRU", 0.246, 0.951, 14.7), ("EEGIFNet", 0.215, 0.968, 14.8),
            ("EEGDiR", 0.278, 0.939, 12.8), ("GCTNet", 0.264, 0.947, 13.6), ("NovelGAN", 0.317, 0.942, 10.6),
            ("DuoCL", 0.309, 0.922, 12.3), ("EEGDNet", 0.270, 0.955, 12.2), ("DeepSeparator", 0.378, 0.923, 8.71),
            ("NovelCNN", 0.347, 0.902, 11.4)]
BANDS = [("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30), ("gamma", 30, 40)]

DONE, SKIPPED = [], []


# ------------------------------------------------------------------ helpers
def save(fig, out, name):
    path = os.path.join(out, name + ".png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    DONE.append(name)


def plot(name, out):
    """Decorator: run one plot, never let one failure stop the others."""
    def deco(fn):
        def run(*a, **k):
            try:
                r = fn(*a, **k)
                if r is False:
                    SKIPPED.append((name, "needs data this run does not have"))
                return r
            except Exception as e:  # noqa: BLE001
                plt.close("all")
                SKIPPED.append((name, "%s: %s" % (type(e).__name__, e)))
        return run
    return deco


def legend_end(ax, x, y, text, color=INK2, dy=0):
    """Direct label next to the last point of a line."""
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=color, fontsize=8, va="center")


class Run:
    """Everything loaded from one run folder."""

    def __init__(self, path, label=None, fs=256, conv="repo"):
        self.path = path
        self.label = label or os.path.basename(os.path.dirname(os.path.abspath(path))) or path
        self.fs, self.conv = fs, conv
        nn = os.path.join(path, "nn_output")
        self.nn = nn
        self.hist = self._load_history()
        self.summary = None
        sj = os.path.join(nn, "metrics_summary.json")
        if os.path.exists(sj):
            with open(sj) as f:
                self.summary = json.load(f)
            run = self.summary.get("run", {})
            self.conv = run.get("snr_convention", conv)
        self.has_test = all(os.path.exists(os.path.join(nn, f)) for f in
                            ("noiseinput_test.npy", "Denoiseoutput_test.npy", "EEG_test.npy"))
        self.snr_levels = np.linspace(-7, 2, 10)
        si = os.path.join(path, "split_info.npz")
        if os.path.exists(si):
            z = np.load(si, allow_pickle=True)
            self.snr_levels = np.asarray(z["snr_levels"], float)
            self.conv = str(z["snr_convention"]) if "snr_convention" in z.files else self.conv
        self.nlev = len(self.snr_levels)
        if self.has_test:
            self.noisy = np.load(os.path.join(nn, "noiseinput_test.npy")).astype(np.float64)
            self.clean = np.load(os.path.join(nn, "EEG_test.npy")).astype(np.float64)
            self.den = np.load(os.path.join(nn, "Denoiseoutput_test.npy")).astype(np.float64)
            n = len(self.clean)
            self.noisy, self.clean, self.den = (a.reshape(n, -1) for a in (self.noisy, self.clean, self.den))
            self.n, self.T = self.clean.shape
            self.fs = 512 if self.T == 1024 else fs
            self.m = self.n // self.nlev
            self.seg = segment_metrics(self.den, self.clean, self.fs)
            self.seg_noisy = segment_metrics(self.noisy, self.clean, self.fs)
            self.block = np.repeat(np.arange(self.nlev), self.m)[: self.n]

    def _load_history(self):
        h = {}
        sp = os.path.join(self.path, "checkpoint", "state.pkl")
        if os.path.exists(sp):
            with open(sp, "rb") as f:
                s = pickle.load(f)
            src = s.get("hist", {})
            for k in ("train_mse", "train_mse_eval", "val_mse", "grads", "val_cc", "val_rrmse", "val_psnr", "val_snr",
                      "val_rrmse_t", "val_rrmse_f", "val_rrmse_s", "val_rrmse_total", "val_rrmse_spectral", "lr"):
                if len(src.get(k, [])):
                    h[k] = np.asarray(src[k], float)
            for old, new in (("train_mse_history", "train_mse"), ("val_mse_history", "val_mse"),
                             ("mse_grads_history", "grads"), ("val_cc_history", "val_cc"),
                             ("val_rrmse_history", "val_rrmse")):
                if new not in h and len(s.get(old, [])):
                    h[new] = np.asarray(s[old], float)
            h["best_epoch"] = s.get("best_epoch")
        lp = os.path.join(self.nn, "loss_history.npy")
        if os.path.exists(lp) and "val_mse" not in h:
            d = np.load(lp, allow_pickle=True).item()
            h["train_mse"] = np.asarray(d["loss"]["train_mse"], float)
            h["val_mse"] = np.asarray(d["loss"]["val_mse"], float)
            if "grads" in d:
                g = d["grads"]
                h["grads"] = np.asarray(g["mse"] if isinstance(g, dict) else g, float)
            h["best_epoch"] = d.get("best_epoch")
        if h.get("val_mse") is not None and not h.get("best_epoch"):
            h["best_epoch"] = int(np.argmin(h["val_mse"])) + 1
        return h

    @property
    def snr_axis_label(self):
        return "Input SNR (dB, nominal; repo convention = 2x by 20·log10)" if self.conv == "repo" else "Input SNR (dB)"

    def per_block(self, arr):
        a = np.asarray(arr)[: self.nlev * self.m].reshape(self.nlev, self.m)
        return a.mean(1), a.std(1)


def epochs_of(v):
    return np.arange(1, len(v) + 1)


def figure_note(fig, text):
    fig.text(0.01, 0.005, text, fontsize=7.5, color=MUTED, ha="left", va="bottom")


# ------------------------------------------------------------------ tables
def write_tables(r, out):
    if not r.has_test:
        return
    rows = [("Noisy input", r.seg_noisy), ("Denoised (model)", r.seg)]
    keys = ["mse", "cc", "rrmse_temporal", "rrmse_spectral", "psnr", "snr", "rrmse_t", "rrmse_f", "rrmse_s", "rrmse_total"]
    with open(os.path.join(out, "metrics_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal"] + keys)
        for name, seg in rows:
            w.writerow([name] + ["%.5f" % np.mean(seg[k]) for k in keys])
    with open(os.path.join(out, "metrics_by_snr_mean_sd.csv"), "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["nominal_snr_db", "n"]
        for k in keys:
            hdr += [k + "_mean", k + "_sd"]
        w.writerow(hdr)
        for i in range(r.nlev):
            row = [r.snr_levels[i], r.m]
            for k in keys:
                mu, sd = r.per_block(r.seg[k])
                row += ["%.5f" % mu[i], "%.5f" % sd[i]]
            w.writerow(row)
        row = ["all", r.n]
        for k in keys:
            row += ["%.5f" % np.mean(r.seg[k]), "%.5f" % np.std(r.seg[k])]
        w.writerow(row)
    with open(os.path.join(out, "table5_vs_paper.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "ours", "paper_ReHA-Net_EOG"])
        for k, v in PAPER_T5.items():
            w.writerow([k, "%.4f" % np.mean(r.seg[k]), v])


# ------------------------------------------------------------------ training behaviour
def p01_loss(r, out):
    h = r.hist
    if "val_mse" not in h:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    e = epochs_of(h["val_mse"])
    ax.plot(e, h["train_mse"], color=BLUE, marker="o", ms=3, label="Training loss (during training)")
    if "train_mse_eval" in h:
        ax.plot(e, h["train_mse_eval"], color=AQUA, marker="o", ms=3, label="Training loss (inference mode)")
    ax.plot(e, h["val_mse"], color=ORANGE, marker="s", ms=3, label="Validation loss")
    be = int(h["best_epoch"])
    ax.plot([be], [h["val_mse"][be - 1]], marker="o", ms=9, mfc="none", mec=INK, mew=1.4, zorder=5)
    last = " (last epoch run: still improving)" if be == len(e) else ""
    ax.text(0.02, 0.04, "circled: best validation, epoch %d%s" % (be, last), transform=ax.transAxes,
            fontsize=8.5, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    save(fig, out, "01_loss_vs_epochs")


def p08_loss20(r, out):
    h = r.hist
    if "val_mse" not in h:
        return False
    n = min(20, len(h["val_mse"]))
    e = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(e, h["train_mse"][:n], color=BLUE, marker="o", label="Training loss")
    ax.plot(e, h["val_mse"][:n], color=ORANGE, marker="s", label="Validation loss")
    ax.set_xticks(e[::2])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss across %d epochs" % n)
    ax.legend()
    save(fig, out, "08_loss_first20_epochs")


def p_metric_epochs(r, out):
    """CC and RRMSE across the first 20 epochs (the two plots plot_metric_epochs.py makes)."""
    h = r.hist
    n = min(20, len(h.get("val_cc", [])))
    if n == 0:
        return False
    e = np.arange(1, n + 1)
    for key, name, ylab, col in (("val_cc", "cc_vs_epochs", "Correlation coefficient (CC)", BLUE),
                                 ("val_rrmse", "rrmse_vs_epochs", "RRMSE", ORANGE)):
        if key not in h:
            continue
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        ax.plot(e, h[key][:n], color=col, marker="o")
        ax.set_xticks(e[::2])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.set_title("%s across %d epochs" % ("CC" if key == "val_cc" else "RRMSE", n))
        save(fig, out, name)


def p21_psnr_snr_epochs(r, out):
    h = r.hist
    if "val_psnr" not in h:
        return False
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    e = epochs_of(h["val_psnr"])
    for ax, k, lab, c in ((axs[0], "val_psnr", "PSNR (dB)", BLUE), (axs[1], "val_snr", "SNR (dB)", ORANGE)):
        ax.plot(e, h[k], color=c, marker="o", ms=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(lab)
        ax.set_title("Validation %s across epochs" % lab.split()[0])
    save(fig, out, "21_psnr_snr_vs_epochs")


def p22_rrmse_variants(r, out):
    h = r.hist
    if "val_rrmse_total" not in h:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    e = epochs_of(h["val_rrmse_total"])
    for k, lab, c in (("val_rrmse_total", "RRMSE (mean of t, f, s)", BLUE), ("val_rrmse_t", "RRMSE-t (time)", ORANGE),
                      ("val_rrmse_s", "RRMSE-s (scaled)", AQUA), ("val_rrmse_f", "RRMSE-f (frequency)", YELLOW)):
        if k in h:
            ax.plot(e, h[k], color=c, label=lab)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RRMSE")
    ax.set_title("Validation RRMSE and its variants across epochs")
    ax.legend()
    save(fig, out, "22_rrmse_variants_vs_epochs")


def p23_grad_lr(r, out):
    h = r.hist
    if "grads" not in h:
        return False
    has_lr = "lr" in h and np.isfinite(h["lr"]).any()
    fig, axs = plt.subplots(1, 2 if has_lr else 1, figsize=(11 if has_lr else 6.2, 4.2), squeeze=False)
    e = epochs_of(h["grads"])
    axs[0, 0].plot(e, h["grads"], color=BLUE, marker="o", ms=3)
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Gradient norm (first layer)")
    axs[0, 0].set_title("Gradient norm per epoch")
    if has_lr:
        axs[0, 1].plot(epochs_of(h["lr"]), h["lr"], color=ORANGE, marker="o", ms=3)
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Learning rate")
        axs[0, 1].set_title("Learning rate per epoch")
    save(fig, out, "23_gradient_norm_and_lr")


def p24_gap(r, out):
    h = r.hist
    if "train_mse_eval" not in h:
        return False
    e = epochs_of(h["val_mse"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(e, h["train_mse_eval"], color=AQUA, label="Training loss (inference mode)")
    axs[0].plot(e, h["val_mse"], color=ORANGE, label="Validation loss")
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss"); axs[0].set_title("Like-for-like train vs validation"); axs[0].legend()
    gap = h["val_mse"] - h["train_mse_eval"]
    axs[1].plot(e, gap, color=BLUE)
    axs[1].axhline(0, color=AXIS, lw=1)
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Validation - training loss"); axs[1].set_title("Generalisation gap")
    save(fig, out, "24_generalisation_gap")


def p20_convergence(r, out, targets=(0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.055, 0.05)):
    h = r.hist
    if "val_mse" not in h:
        return False
    v = h["val_mse"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(epochs_of(v), v, color=BLUE, marker="o", ms=3)
    reach = []
    for t in targets:
        idx = np.where(v <= t)[0]
        axs[0].axhline(t, color=GRID, lw=0.8)
        if len(idx):
            axs[0].plot([idx[0] + 1], [v[idx[0]]], marker="o", ms=7, mfc="none", mec=INK)
        reach.append(idx[0] + 1 if len(idx) else None)
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Validation loss"); axs[0].set_title("Validation loss and target levels")
    xs = [str(t) for t, r_ in zip(targets, reach) if r_ is not None]
    ys = [r_ for r_ in reach if r_ is not None]
    axs[1].bar(xs, ys, color=BLUE, width=0.6)
    for x, y in zip(xs, ys):
        axs[1].text(x, y + 0.4, str(y), ha="center", fontsize=8.5, color=INK2)
    axs[1].set_xlabel("Target validation loss"); axs[1].set_ylabel("First epoch at or below target")
    axs[1].set_title("Epochs needed to reach each target")
    axs[1].grid(axis="x", visible=False)
    save(fig, out, "20_convergence_thresholds")


# ------------------------------------------------------------------ final performance
def pick_median(r, blk):
    idx = np.where(r.block == blk)[0]
    c = r.seg["cc"][idx]
    return idx[np.argsort(c)[len(c) // 2]]


def p02_waveforms(r, out):
    if not r.has_test:
        return False
    t = np.arange(r.T) / r.fs
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)
    for ax, blk in zip(axs, (0, r.nlev - 1)):
        i = pick_median(r, blk)
        ax.plot(t, r.noisy[i], color=NOISY_C, lw=1, alpha=0.6, label="Noisy input")
        ax.plot(t, r.clean[i], color=CLEAN_C, lw=1.3, label="Clean EEG (target)")
        ax.plot(t, r.den[i], color=MODEL_C, lw=1.3, label="Model output")
        ax.set_title("SNR = %+g dB  (median-quality segment, CC = %.2f)" % (r.snr_levels[blk], r.seg["cc"][i]))
        ax.set_ylabel("Amplitude (standardised)")
    axs[0].legend(ncol=3, loc="upper right")
    axs[1].set_xlabel("Time (s)")
    save(fig, out, "02_waveform_comparison")


def mean_psd(x, fs, fmax=120):
    return psd(x, fs, fmax).mean(0)


def p03_psd(r, out):
    if not r.has_test:
        return False
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, sl, title in ((axs[0], slice(None), "All test segments"),
                          (axs[1], slice(0, r.m), "Worst noise level (SNR = %+g dB)" % r.snr_levels[0])):
        n_, c_, d_ = r.noisy[sl], r.clean[sl], r.den[sl]
        pn, pc, pd_ = (mean_psd(a, r.fs, 80) for a in (n_, c_, d_))
        f = np.linspace(0, 80, len(pc))
        ax.semilogy(f, pn, color=NOISY_C, lw=1.4, label="Noisy input")
        ax.semilogy(f, pc, color=CLEAN_C, lw=2, label="Clean EEG")
        ax.semilogy(f, pd_, color=MODEL_C, lw=1.8, ls="--", label="Model output")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(title)
    axs[0].set_ylim(min(pc.min(), pd_.min()) * 0.3, pn.max() * 3)
    axs[0].set_ylabel("Mean power spectral density")
    axs[0].legend()
    save(fig, out, "03_psd")


def p04_metrics_vs_snr(r, out):
    if not r.has_test:
        return False
    panels = [("rrmse_temporal", "RRMSE (time)", False), ("rrmse_spectral", "RRMSE (spectral)", True),
              ("cc", "Correlation coefficient", False), ("psnr", "PSNR (dB)", False),
              ("snr", "Output SNR (dB)", False), ("rrmse_total", "RRMSE (mean of t, f, s)", False)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (k, lab, log) in zip(axs.ravel(), panels):
        for seg, col, name in ((r.seg_noisy, NOISY_C, "Noisy input"), (r.seg, MODEL_C, "Model output")):
            mu, sd = r.per_block(seg[k])
            ax.plot(r.snr_levels, mu, color=col, marker="o", label=name)
            ax.fill_between(r.snr_levels, mu - sd, mu + sd, color=col, alpha=0.13, lw=0)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel(r.snr_axis_label if k in ("cc", "snr", "rrmse_total") else "Input SNR (dB)")
        ax.set_ylabel(lab)
    axs[0, 0].legend()
    fig.suptitle("Performance vs noise level (mean ± 1 SD across test segments)")
    fig.tight_layout()
    save(fig, out, "04_metrics_vs_snr")


def p05_boxplots(r, out):
    if not r.has_test:
        return False
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (k, lab) in zip(axs, (("rrmse_temporal", "RRMSE (time)"), ("cc", "Correlation coefficient"), ("psnr", "PSNR (dB)"))):
        data = [r.seg[k][r.block == i] for i in range(r.nlev)]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6,
                        medianprops=dict(color=ORANGE, lw=1.6), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                        boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE))
        ax.set_xticklabels(["%+g" % s for s in r.snr_levels])
        ax.set_xlabel("Input SNR (dB)")
        ax.set_ylabel(lab)
    fig.suptitle("Spread of results across test segments (outliers hidden)")
    fig.tight_layout()
    save(fig, out, "05_boxplots")


def p06_pred_true(r, out):
    if not r.has_test:
        return False
    x, y = r.clean.ravel(), r.den.ravel()
    lim = np.percentile(np.abs(x), 99.5)
    r2 = 1 - np.sum((y - x) ** 2) / np.sum((x - x.mean()) ** 2)
    fig, ax = plt.subplots(figsize=(6, 5.6))
    hb = ax.hexbin(x, y, gridsize=70, bins="log", cmap=SEQ, extent=(-lim, lim, -lim, lim), mincnt=1)
    ax.plot([-lim, lim], [-lim, lim], color=ORANGE, lw=1.2, label="perfect (y = x)")
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("True clean EEG"); ax.set_ylabel("Model output")
    ax.set_title("Predicted vs true  (R² = %.2f)" % r2)
    ax.legend(loc="upper left")
    fig.colorbar(hb, label="count (log scale)", shrink=0.8)
    save(fig, out, "06_predicted_vs_true")


def p07_residuals(r, out):
    if not r.has_test:
        return False
    i = pick_median(r, r.nlev - 1)
    res = r.den - r.clean
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    axs[0].plot(np.arange(r.T) / r.fs, res[i], color=VIOLET, lw=1.2)
    axs[0].axhline(0, color=AXIS, lw=1)
    axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Residual"); axs[0].set_title("Residual (output - clean), example at SNR = %+g dB" % r.snr_levels[-1])
    axs[1].hist(res.ravel(), bins=160, range=(-1.5, 1.5), density=True, color=BLUE)
    axs[1].set_xlabel("Residual"); axs[1].set_ylabel("Density")
    axs[1].set_title("All residuals: mean = %+.4f, SD = %.3f" % (res.mean(), res.std()))
    save(fig, out, "07_residuals")


def p09_spectral_diff(r, out):
    if not r.has_test:
        return False
    pc = mean_psd(r.clean, r.fs, 40)
    f = np.linspace(0, 40, len(pc))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(f, 10 * np.log10(mean_psd(r.noisy, r.fs, 40) / pc), color=NOISY_C, ls="--", label="Noisy input")
    ax.plot(f, 10 * np.log10(mean_psd(r.den, r.fs, 40) / pc), color=MODEL_C, label="Model output")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power difference vs clean (dB)")
    ax.set_title("Spectral power difference (signal - clean), 0-40 Hz")
    ax.legend()
    save(fig, out, "09_spectral_difference")


def grouped_bars(ax, labels, a, b, la, lb, fmt="%.3g"):
    x = np.arange(len(labels))
    ax.bar(x - 0.2, a, 0.38, color=BLUE, label=la)
    ax.bar(x + 0.2, b, 0.38, color=ORANGE, label=lb)
    for xi, va, vb in zip(x, a, b):
        ax.text(xi - 0.2, va, fmt % va, ha="center", va="bottom", fontsize=8, color=INK2)
        ax.text(xi + 0.2, vb, fmt % vb, ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.grid(axis="x", visible=False)


def p10_vs_paper(r, out):
    if not r.has_test:
        return False
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [2, 1, 4]})
    ours = {k: float(np.mean(r.seg[k])) for k in PAPER_T5}
    for ax, ks, lab, labels in ((axs[0], ["psnr", "snr"], "dB (higher is better)", ["PSNR", "SNR"]),
                                (axs[1], ["cc"], "CC (higher is better)", ["CC"]),
                                (axs[2], ["rrmse_total", "rrmse_s", "rrmse_t", "rrmse_f"], "RRMSE (lower is better)",
                                 ["RRMSE", "RRMSE-s", "RRMSE-t", "RRMSE-f"])):
        grouped_bars(ax, labels, [ours[k] for k in ks], [PAPER_T5[k] for k in ks], "Ours (this run)", "ReHA-Net paper (EOG)")
        ax.set_ylabel(lab)
        ax.set_ylim(0, max(max(ours[k], PAPER_T5[k]) for k in ks) * 1.2)
    axs[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Ours vs ReHA-Net (paper Table 5). Different data range and model: read as a gap, not a like-for-like test")
    fig.tight_layout()
    save(fig, out, "10_metrics_vs_paper")


def p11_model_comparison(r, out):
    if not r.has_test:
        return False
    ours = (float(np.mean(r.seg["rrmse_temporal"])), float(np.mean(r.seg["cc"])), float(np.mean(r.seg["snr"])))
    rows = [(n, a, b, c) for n, a, b, c in PAPER_T4] + [("This run", *ours)]
    names = [x[0] for x in rows]
    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, j, lab in ((axs[0], 1, "RRMSE (lower is better)"), (axs[1], 2, "CC (higher is better)"), (axs[2], 3, "SNR dB (higher is better)")):
        vals = [x[j] for x in rows]
        cols = [BLUE if n == "ReHA-Net" else (ORANGE if n == "This run" else "#c9c8c1") for n in names]
        ax.bar(names, vals, color=cols, width=0.65)
        for i, v in enumerate(vals):
            ax.text(i, v, "%.3g" % v, ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
    axs[1].set_ylim(0.85, 1.0)
    plt.setp(axs[2].get_xticklabels(), rotation=30, ha="right")
    fig.suptitle("EOG artifact removal: paper Table 4 values vs this run (ours is on the -7..+2 dB data; read as a gap)")
    fig.tight_layout()
    save(fig, out, "11_model_comparison_bars")


def p12_cdf(r, out):
    if not r.has_test:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.4))
    e = r.seg["rrmse_temporal"]
    sets = [("All segments", e, INK), ("%+g dB (hardest)" % r.snr_levels[0], e[r.block == 0], BLUE),
            ("%+g dB" % r.snr_levels[r.nlev // 2], e[r.block == r.nlev // 2], ORANGE),
            ("%+g dB (easiest)" % r.snr_levels[-1], e[r.block == r.nlev - 1], AQUA)]
    for lab, v, c in sets:
        v = np.sort(v)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, label=lab, lw=1.6)
    ax.set_xlabel("Per-segment RRMSE (time)"); ax.set_ylabel("Fraction of segments with error ≤ x")
    ax.set_title("Error distribution (CDF)")
    ax.legend(loc="lower right")
    save(fig, out, "12_error_cdf")


def p13_pass_rate(r, out):
    if not r.has_test:
        return False
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    th = np.linspace(0.5, 0.99, 100)
    axs[0].plot(th, [100 * np.mean(r.seg["cc"] >= t) for t in th], color=MODEL_C, label="Model")
    axs[0].plot(th, [100 * np.mean(r.seg_noisy["cc"] >= t) for t in th], color=NOISY_C, label="Noisy input")
    axs[0].set_xlabel("CC threshold"); axs[0].set_ylabel("% of test segments with CC ≥ threshold"); axs[0].legend()
    th = np.linspace(0.1, 1.5, 100)
    axs[1].plot(th, [100 * np.mean(r.seg["rrmse_temporal"] <= t) for t in th], color=MODEL_C, label="Model")
    axs[1].plot(th, [100 * np.mean(r.seg_noisy["rrmse_temporal"] <= t) for t in th], color=NOISY_C, label="Noisy input")
    axs[1].set_xlabel("RRMSE (time) threshold"); axs[1].set_ylabel("% of test segments with RRMSE ≤ threshold"); axs[1].legend()
    fig.suptitle("Pass-rate curves")
    save(fig, out, "13_pass_rate_curves")


def p14_error_vs_amp(r, out):
    if not r.has_test:
        return False
    a = rms(r.clean)
    e = r.seg["rrmse_temporal"]
    qs = np.quantile(a, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(a, qs[1:-1]), 0, 9)
    x = [a[idx == i].mean() for i in range(10)]
    med = [np.median(e[idx == i]) for i in range(10)]
    lo = [np.percentile(e[idx == i], 25) for i in range(10)]
    hi = [np.percentile(e[idx == i], 75) for i in range(10)]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(x, med, color=MODEL_C, marker="o", label="median")
    ax.fill_between(x, lo, hi, color=MODEL_C, alpha=0.15, lw=0, label="25-75%")
    ax.set_xlabel("Clean-segment RMS (standardised units, decile means)"); ax.set_ylabel("RRMSE (time)")
    ax.set_title("Error vs signal amplitude")
    ax.legend()
    save(fig, out, "14_error_vs_amplitude")


def p15_best_worst(r, out):
    if not r.has_test:
        return False
    c = r.seg["cc"]
    best, worst = np.argsort(-c)[:3], np.argsort(c)[:3]
    t = np.arange(r.T) / r.fs
    fig, axs = plt.subplots(2, 3, figsize=(15, 6.4), sharex=True)
    for row, (ids, nm) in enumerate(((best, "Best"), (worst, "Worst"))):
        for col, i in enumerate(ids):
            ax = axs[row, col]
            ax.plot(t, r.noisy[i], color=NOISY_C, lw=1, alpha=0.6, label="Noisy")
            ax.plot(t, r.clean[i], color=CLEAN_C, lw=1.2, label="Clean")
            ax.plot(t, r.den[i], color=MODEL_C, lw=1.2, label="Model")
            ax.set_title("%s: SNR %+g dB, CC %.2f" % (nm, r.snr_levels[r.block[i]], c[i]), fontsize=9.5)
    axs[0, 0].legend(fontsize=8)
    fig.supxlabel("Time (s)")
    fig.suptitle("Best and worst test segments by CC")
    fig.tight_layout()
    save(fig, out, "15_best_worst_segments")


def p16_heatmap(r, out):
    if not r.has_test:
        return False
    e = r.seg["rrmse_temporal"][: r.nlev * r.m].reshape(r.nlev, r.m)
    order = np.argsort(e.mean(0))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.imshow(e[:, order], aspect="auto", origin="lower", cmap=SEQ, interpolation="nearest")
    ax.set_yticks(range(r.nlev)); ax.set_yticklabels(["%+g" % s for s in r.snr_levels])
    ax.grid(False)
    ax.set_xlabel("Test segments (sorted by mean error over SNR levels)"); ax.set_ylabel("Nominal input SNR (dB)")
    ax.set_title("RRMSE (time): SNR × segment")
    fig.colorbar(im, label="RRMSE")
    save(fig, out, "16_snr_segment_heatmap")


def band_filter(x, fs, lo, hi):
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(x.shape[-1], 1.0 / fs)
    X[..., (f < lo) | (f >= hi)] = 0
    return np.fft.irfft(X, n=x.shape[-1], axis=-1)


def p17_band_cc(r, out):
    if not r.has_test:
        return False
    nb, mb = [], []
    for _, lo, hi in BANDS:
        c = band_filter(r.clean, r.fs, lo, hi)
        nb.append(np.nanmean(cc(band_filter(r.noisy, r.fs, lo, hi), c)))
        mb.append(np.nanmean(cc(band_filter(r.den, r.fs, lo, hi), c)))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    grouped_bars(ax, ["%s\n%d-%d Hz" % b for b in BANDS], nb, mb, "Noisy input", "Model", fmt="%.2f")
    ax.containers[0].patches and [p.set_color(NOISY_C) for p in ax.containers[0].patches]
    [p.set_color(MODEL_C) for p in ax.containers[1].patches]
    ax.legend()
    ax.set_ylabel("CC with clean EEG in that band"); ax.set_ylim(0, 1.08)
    ax.set_title("Per-frequency-band correlation")
    save(fig, out, "17_per_band_cc")


def p18_classical(r, out):
    if not r.has_test:
        return False
    try:
        from scipy.signal import butter, sosfiltfilt
    except Exception:
        SKIPPED.append(("18_classical_baseline", "scipy not installed"))
        return None
    cuts = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
    res = {}
    for fc in cuts:
        sos = butter(4, fc, btype="highpass", fs=r.fs, output="sos")
        res[fc] = sosfiltfilt(sos, r.noisy, axis=-1)
    means = [np.mean(cc(res[fc], r.clean)) for fc in cuts]
    best = cuts[int(np.argmax(means))]
    hp_cc = cc(res[best], r.clean)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    axs[0].plot(cuts, means, color=ORANGE, marker="o", label="High-pass filter (4th-order Butterworth)")
    axs[0].axhline(np.mean(r.seg["cc"]), color=MODEL_C, label="Model")
    axs[0].axhline(np.mean(r.seg_noisy["cc"]), color=NOISY_C, ls="--", label="Noisy input")
    axs[0].set_xlabel("Cut-off (Hz)"); axs[0].set_ylabel("Mean CC")
    axs[0].set_title("Cut-off sweep (best %g Hz, picked ON the test set, so it favours the filter)" % best, fontsize=9.5)
    axs[0].legend(fontsize=8)
    x = np.arange(r.nlev)
    for off, v, col, lab in ((-0.27, r.per_block(r.seg_noisy["cc"])[0], NOISY_C, "Noisy input"),
                             (0, r.per_block(hp_cc)[0], ORANGE, "High-pass %g Hz" % best),
                             (0.27, r.per_block(r.seg["cc"])[0], MODEL_C, "Model")):
        axs[1].bar(x + off, v, 0.26, color=col, label=lab)
    axs[1].set_xticks(x); axs[1].set_xticklabels(["%+g" % s for s in r.snr_levels])
    axs[1].set_xlabel("Nominal input SNR (dB)"); axs[1].set_ylabel("Mean CC"); axs[1].set_ylim(0, 1.1)
    axs[1].legend(fontsize=8, loc="lower right"); axs[1].grid(axis="x", visible=False)
    fig.suptitle("Classical baseline: high-pass filter vs the network")
    save(fig, out, "18_classical_baseline")


def p19_data_overview(r, out):
    if not r.has_test:
        return False
    t = np.arange(r.T) / r.fs
    seg = 0
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    ax = axs[0, 0]
    ax.plot(t, r.clean[seg] + 0, color=INK, lw=1)
    for k in range(r.nlev):
        i = k * r.m + seg
        ax.plot(t, r.noisy[i] - (k + 1) * 6, color=MUTED, lw=0.9)
    ax.set_yticks([0] + [-(k + 1) * 6 for k in range(r.nlev)])
    ax.set_yticklabels(["clean"] + ["%+g dB" % s for s in r.snr_levels][::1])
    ax.grid(False)
    ax.set_xlabel("Time (s)"); ax.set_title("One segment mixed at each nominal SNR level")
    ax = axs[0, 1]
    noise = r.noisy - r.clean
    f = np.linspace(0, 80, len(mean_psd(r.clean, r.fs, 80)))
    ax.semilogy(f, mean_psd(r.clean, r.fs, 80), color=INK, label="Clean EEG")
    ax.semilogy(f, mean_psd(noise, r.fs, 80), color=BLUE, label="Added noise (EOG)")
    ax.semilogy(f, mean_psd(r.noisy, r.fs, 80), color=MUTED, ls="--", label="Noisy input")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Mean PSD"); ax.set_title("Signal vs noise spectra"); ax.legend()
    ax = axs[1, 0]
    ax.hist(r.clean.ravel(), bins=160, range=(-6, 6), density=True, alpha=0.6, color=BLUE, label="Clean")
    ax.hist(r.noisy.ravel(), bins=160, range=(-6, 6), density=True, alpha=0.5, color=ORANGE, label="Noisy")
    ax.set_xlabel("Amplitude (standardised)"); ax.set_ylabel("Density"); ax.set_title("Amplitude histograms"); ax.legend()
    ax = axs[1, 1]
    meas = [np.mean(measured_snr_db(r.clean[k * r.m:(k + 1) * r.m], r.noisy[k * r.m:(k + 1) * r.m], "standard")) for k in range(r.nlev)]
    ax.plot(r.snr_levels, meas, color=ORANGE, marker="o", label="Measured 20·log10(RMS clean / RMS noise)")
    ax.plot(r.snr_levels, r.snr_levels, color=INK, ls="--", lw=1, label="y = x")
    ax.set_xlabel("Nominal SNR (dB)"); ax.set_ylabel("Measured SNR (dB)"); ax.set_title("Nominal vs measured SNR"); ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out, "19_data_overview")


def p27_snr_gain(r, out):
    """How many dB the network adds at each input level (PSNR/SNR out minus in)."""
    if not r.has_test:
        return False
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, k, lab in ((axs[0], "snr", "Output SNR (dB)"), (axs[1], "psnr", "PSNR (dB)")):
        for seg, col, nm in ((r.seg_noisy, NOISY_C, "Noisy input"), (r.seg, MODEL_C, "Model output")):
            mu, sd = r.per_block(seg[k])
            ax.plot(r.snr_levels, mu, color=col, marker="o", label=nm)
            ax.fill_between(r.snr_levels, mu - sd, mu + sd, color=col, alpha=0.13, lw=0)
        ax.set_xlabel("Input SNR (dB)"); ax.set_ylabel(lab)
    axs[0].legend()
    fig.suptitle("PSNR and SNR before and after denoising")
    save(fig, out, "27_psnr_snr_vs_input_snr")


# ------------------------------------------------------------------ comparison of several runs
def compare(runs, out, targets=(0.1, 0.08, 0.07, 0.06, 0.055, 0.05)):
    os.makedirs(out, exist_ok=True)
    cols = SERIES[: len(runs)]
    if len(runs) > len(SERIES):
        raise SystemExit("compare supports at most %d runs" % len(SERIES))

    def lines(name, key, ylab, title, log=False, first=None):
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        ok = False
        for r, c in zip(runs, cols):
            if key in r.hist:
                v = r.hist[key][:first] if first else r.hist[key]
                ax.plot(epochs_of(v), v, color=c, label=r.label)
                ok = True
        if not ok:
            plt.close(fig)
            SKIPPED.append((name, "no run has %s" % key))
            return
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylab); ax.set_title(title); ax.legend()
        save(fig, out, name)

    lines("c01_validation_loss", "val_mse", "Validation loss", "Validation loss", log=True)
    lines("c02_validation_loss_first20", "val_mse", "Validation loss", "Validation loss, first 20 epochs", first=20)
    lines("c03_training_loss_inference", "train_mse_eval", "Training loss", "Training loss (inference mode)", log=True)
    lines("c04_val_cc", "val_cc", "Validation CC", "Validation CC across epochs")
    lines("c05_val_rrmse_total", "val_rrmse_total", "Validation RRMSE (mean of t, f, s)", "Validation RRMSE across epochs")
    lines("c06_val_psnr", "val_psnr", "Validation PSNR (dB)", "Validation PSNR across epochs")
    lines("c07_lr", "lr", "Learning rate", "Learning rate across epochs")

    # epochs needed to reach loss targets
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    w = 0.8 / len(runs)
    for j, (r, c) in enumerate(zip(runs, cols)):
        v = r.hist.get("val_mse")
        if v is None:
            continue
        ys = []
        for t in targets:
            idx = np.where(v <= t)[0]
            ys.append(idx[0] + 1 if len(idx) else np.nan)
        x = np.arange(len(targets)) + (j - (len(runs) - 1) / 2) * w
        ax.bar(x, np.nan_to_num(ys), w * 0.92, color=c, label=r.label)
        for xi, y in zip(x, ys):
            ax.text(xi, 0 if np.isnan(y) else y, "n/r" if np.isnan(y) else "%d" % y, ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(targets))); ax.set_xticklabels([str(t) for t in targets])
    ax.set_xlabel("Target validation loss"); ax.set_ylabel("First epoch at or below target (n/r = not reached)")
    ax.set_title("Training efficiency: epochs to reach each loss target"); ax.legend(); ax.grid(axis="x", visible=False)
    save(fig, out, "c08_epochs_to_target")

    # final test metrics
    have = [r for r in runs if r.has_test]
    if have:
        mets = [("cc", "CC"), ("rrmse_temporal", "RRMSE (time)"), ("rrmse_total", "RRMSE (mean t,f,s)"), ("psnr", "PSNR (dB)"), ("snr", "SNR (dB)"),
                ("rrmse_spectral", "RRMSE (spectral)")]
        fig, axs = plt.subplots(2, 3, figsize=(14, 7.4))
        for ax, (k, lab) in zip(axs.ravel(), mets):
            vals = [float(np.mean(r.seg[k])) for r in have]
            ax.bar(range(len(have)), vals, color=[cols[runs.index(r)] for r in have], width=0.6)
            for i, v in enumerate(vals):
                ax.text(i, v, "%.4g" % v, ha="center", va="bottom", fontsize=8, color=INK2)
            ax.set_xticks(range(len(have))); ax.set_xticklabels([r.label for r in have], rotation=20, ha="right", fontsize=8)
            ax.set_ylabel(lab); ax.grid(axis="x", visible=False)
            lo, hi = min(vals), max(vals)
            if k == "cc":
                ax.set_ylim(max(0, lo - 0.05), min(1.0, hi + 0.02))
        fig.suptitle("Final test metrics")
        fig.tight_layout()
        save(fig, out, "c09_final_metrics")

        fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
        for ax, (k, lab) in zip(axs, (("cc", "CC"), ("rrmse_temporal", "RRMSE (time)"), ("psnr", "PSNR (dB)"))):
            for r in have:
                mu, _ = r.per_block(r.seg[k])
                ax.plot(r.snr_levels, mu, color=cols[runs.index(r)], marker="o", label=r.label)
            ax.set_xlabel("Input SNR (dB)"); ax.set_ylabel(lab)
        axs[0].legend(fontsize=8)
        fig.suptitle("Performance by noise level, all runs (low SNR is where models differ)")
        save(fig, out, "c10_metrics_vs_snr")

        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        base = have[0]
        for r in have:
            mu, _ = r.per_block(r.seg["rrmse_temporal"])
            ax.plot(r.snr_levels, mu, color=cols[runs.index(r)], marker="o", label=r.label)
        ax.set_xlabel("Input SNR (dB)"); ax.set_ylabel("RRMSE (time)"); ax.set_title("RRMSE by noise level, all runs"); ax.legend()
        save(fig, out, "c11_rrmse_vs_snr")

        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        pc = mean_psd(have[0].clean, have[0].fs, 40)
        f = np.linspace(0, 40, len(pc))
        for r in have:
            ax.plot(f, 10 * np.log10(mean_psd(r.den, r.fs, 40) / pc), color=cols[runs.index(r)], label=r.label)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Output power vs clean (dB)")
        ax.set_title("Spectral bias, all runs (closer to 0 is better; spectral loss should help here)"); ax.legend()
        save(fig, out, "c12_spectral_bias")

        with open(os.path.join(out, "compare_table.csv"), "w", newline="") as f_:
            w_ = csv.writer(f_)
            w_.writerow(["run", "epochs_run", "best_epoch", "cc", "rrmse_temporal", "rrmse_total", "psnr", "snr", "rrmse_spectral", "cc_at_lowest_snr"])
            for r in have:
                h = r.hist
                w_.writerow([r.label, len(h.get("val_mse", [])), h.get("best_epoch")] +
                            ["%.4f" % np.mean(r.seg[k]) for k in ("cc", "rrmse_temporal", "rrmse_total", "psnr", "snr", "rrmse_spectral")] +
                            ["%.4f" % r.per_block(r.seg["cc"])[0][0]])


# ------------------------------------------------------------------ main
SINGLE = [p01_loss, p08_loss20, p_metric_epochs, p20_convergence, p21_psnr_snr_epochs, p22_rrmse_variants, p23_grad_lr, p24_gap,
          p02_waveforms, p03_psd, p04_metrics_vs_snr, p05_boxplots, p06_pred_true, p07_residuals, p09_spectral_diff,
          p10_vs_paper, p11_model_comparison, p12_cdf, p13_pass_rate, p14_error_vs_amp, p15_best_worst, p16_heatmap,
          p17_band_cc, p18_classical, p19_data_overview, p27_snr_gain]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="run folder: results/<foldername>/1")
    ap.add_argument("--out", help="output folder (default <run>/plots, or ./compare for --compare)")
    ap.add_argument("--fs", type=float, default=256)
    ap.add_argument("--snr-convention", choices=["repo", "standard"], default="repo")
    ap.add_argument("--compare", nargs="+", metavar="RUN", help="several run folders to overlay")
    ap.add_argument("--labels", nargs="+", help="names for --compare runs")
    a = ap.parse_args()

    if a.compare:
        labels = a.labels or [os.path.basename(os.path.dirname(os.path.abspath(p))) for p in a.compare]
        if len(labels) != len(a.compare):
            raise SystemExit("--labels needs one name per run")
        runs = [Run(p, l, a.fs, a.snr_convention) for p, l in zip(a.compare, labels)]
        compare(runs, a.out or "compare")
        out = a.out or "compare"
    else:
        if not a.run:
            ap.error("give a run folder, or use --compare")
        r = Run(a.run, None, a.fs, a.snr_convention)
        out = a.out or os.path.join(a.run, "plots")
        os.makedirs(out, exist_ok=True)
        write_tables(r, out)
        for fn in SINGLE:
            wrapped = plot(fn.__name__, out)(fn)
            wrapped(r, out)
    print("\nwrote %d plots to %s" % (len(DONE), os.path.abspath(out)))
    for n in DONE:
        print("  ", n + ".png")
    if SKIPPED:
        print("\nskipped:")
        for n, why in SKIPPED:
            print("  %s -> %s" % (n, why))


if __name__ == "__main__":
    main()
