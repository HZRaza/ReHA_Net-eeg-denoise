"""Phase 2 plots: EOG vs EMG split, gain by input SNR, band-power bias (the novelty evidence), paper comparison.

One run   :  python plots_phase2.py "<results>/pf_MIXED_seed0/1"
Several   :  python plots_phase2.py --compare "<results>/reha_MIXED_baseline/1" "<results>/pf_MIXED_seed0/1" \
                 --labels "v1 (first run)" "paper-faithful" --out compare_v1_pf
Writes PNGs + CSV tables to <run>/plots_phase2 (or --out).  Run plot_results.py for the generic 27 plots first;
this file adds what is specific to Phase 2 (6 SNR levels, EOG/EMG, band-power bias).
Needs numpy + matplotlib.  Reads nn_output/*.npy, split_info.npz and metrics_summary.json from the run folder.
All signals are on the standardised scale (noisy std = 1); every quantity below is a ratio, so scale cancels.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_results as P  # noqa: E402  (style, Run loader, save helper)
from eeg_utils import measured_snr_db  # noqa: E402

plt = P.plt
BANDS = [("delta 1-4", 1, 4), ("theta 4-8", 4, 8), ("alpha 8-13", 8, 13), ("beta 13-30", 13, 30), ("gamma 30-40", 30, 40)]
NOISE = {0: ("EOG", P.BLUE), 1: ("EMG", P.ORANGE)}
# paper Table 5 (EOG, EMG): PSNR, SNR, CC, RRMSE
PAPER = {0: dict(psnr=27.10, snr=17.06, cc=0.976, rrmse_total=0.1653), 1: dict(psnr=26.91, snr=16.87, cc=0.976, rrmse_total=0.1718)}


def note(fig, text):
    fig.text(0.01, -0.035, text, fontsize=7.5, color=P.MUTED, ha="left", va="top")


# ------------------------------------------------------------------ loading
class R2:
    def __init__(self, path, label=None, fs=256):
        self.r = P.Run(path, label, fs, "standard")
        r = self.r
        if not r.has_test:
            raise SystemExit("no test npy files in %s/nn_output" % path)
        z = np.load(os.path.join(path, "split_info.npz"), allow_pickle=True)
        self.lab = np.asarray(z["test_noise_label"]).astype(int)
        if len(self.lab) != r.n:
            raise SystemExit("noise labels (%d) do not match test rows (%d)" % (len(self.lab), r.n))
        self.levels = r.snr_levels
        self.nlev, self.m, self.fs = r.nlev, r.m, r.fs
        self.block = r.block
        self.kinds = [k for k in (0, 1) if (self.lab == k).any()]
        self.label = r.label
        self.in_snr = measured_snr_db(r.clean, r.noisy, "standard")       # per row, real dB
        self.gain = r.seg["snr"] - self.in_snr
        self._bp = None

    def sel(self, k=None, b=None):
        s = np.ones(self.r.n, bool)
        if k is not None:
            s &= self.lab == k
        if b is not None:
            s &= self.block == b
        return s

    def band_power(self):
        """(n, n_bands) band power of clean, output, noisy (rfft energy)."""
        if self._bp is None:
            f = np.fft.rfftfreq(self.r.T, 1.0 / self.fs)
            out = {}
            for name, x in (("clean", self.r.clean), ("den", self.r.den), ("noisy", self.r.noisy)):
                e = np.abs(np.fft.rfft(x, axis=-1)) ** 2
                out[name] = np.stack([e[:, (f >= lo) & (f < hi)].sum(1) for _, lo, hi in BANDS], 1) + 1e-20
            self._bp = out
        return self._bp

    def band_bias(self, sel):
        """mean over segments of 10*log10(P_out / P_clean) per band; negative = output too weak."""
        bp = self.band_power()
        return (10 * np.log10(bp["den"][sel] / bp["clean"][sel])).mean(0)


def lvl_labels(R):
    return ["%g" % v for v in R.levels]


def mean_sd(v, R, k):
    mu = np.array([v[R.sel(k, b)].mean() for b in range(R.nlev)])
    sd = np.array([v[R.sel(k, b)].std() for b in range(R.nlev)])
    return mu, sd


# ------------------------------------------------------------------ single-run plots
def p30_by_noise(R, out):
    panels = [("cc", "CC", None), ("rrmse_t", "RRMSE (temporal)", None), ("snr", "Output SNR (dB)", None),
              ("gain", "SNR gain = output - input (dB)", 0)]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    x = np.arange(R.nlev)
    for ax, (key, title, zero) in zip(axs.ravel(), panels):
        v = R.gain if key == "gain" else R.r.seg[key]
        for k in R.kinds:
            mu, sd = mean_sd(v, R, k)
            ax.errorbar(x + (k - 0.5) * 0.06, mu, sd, color=NOISE[k][1], marker="o", capsize=2, lw=1.6, label=NOISE[k][0])
        if zero is not None:
            ax.axhline(zero, color=P.INK2, lw=1, ls="--")
        ax.set_xticks(x, lvl_labels(R))
        ax.set_xlabel("Input SNR (dB)")
        ax.set_title(title)
    axs[0, 0].legend()
    note(fig, "mean +/- SD over test segments; EOG and EMG are the same model, scored on their own segments")
    P.save(fig, out, "30_metrics_by_noise_vs_snr")


def p31_gain(R, out):
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    x = np.arange(R.nlev)
    for k in R.kinds:
        mu, _ = mean_sd(R.gain, R, k)
        ax.plot(x, mu, marker="o", color=NOISE[k][1], label=NOISE[k][0])
        for xi, yi in zip(x, mu):
            if yi < 0:
                ax.annotate("%.1f" % yi, (xi, yi), xytext=(0, 9 if k else -14), textcoords="offset points", ha="center",
                            color=NOISE[k][1], fontsize=8)
    ax.axhline(0, color=P.INK2, lw=1, ls="--")
    ax.fill_between([-0.4, R.nlev - 0.6], ax.get_ylim()[0], 0, color=P.RED, alpha=0.06, lw=0)
    ax.set_xlim(-0.4, R.nlev - 0.6)
    ax.set_xticks(x, lvl_labels(R))
    ax.set_xlabel("Input SNR (dB)")
    ax.set_ylabel("Output SNR - input SNR (dB)")
    ax.set_title("Does the model help? Below zero it makes the signal worse than doing nothing")
    ax.legend()
    P.save(fig, out, "31_snr_gain_vs_input_snr")


def p32_band_bias(R, out):
    """Novelty evidence: per-band power bias of the output (dB), one panel per noise type, one line per SNR level."""
    fig, axs = plt.subplots(1, len(R.kinds), figsize=(5.2 * len(R.kinds), 4.2), constrained_layout=True, sharey=True,
                            squeeze=False)
    x = np.arange(len(BANDS))
    cm = P.SEQ(np.linspace(0.2, 1, R.nlev))
    for ax, k in zip(axs[0], R.kinds):
        for b in range(R.nlev):
            ax.plot(x, R.band_bias(R.sel(k, b)), marker="o", color=cm[b], label="%g dB" % R.levels[b])
        ax.axhline(0, color=P.INK2, lw=1, ls="--")
        ax.set_xticks(x, [n.replace(" ", "\n") for n, _, _ in BANDS])
        ax.set_title(NOISE[k][0] + " segments")
        ax.set_xlabel("EEG band (Hz)")
    axs[0, 0].set_ylabel("Output power - clean power (dB)\nnegative = output too weak")
    axs[0, -1].legend(title="input SNR", fontsize=7.5, ncol=2)
    note(fig, "mean over segments of 10*log10(P_out/P_clean) in each band")
    P.save(fig, out, "32_band_power_bias")


def p33_band_vs_snr(R, out):
    fig, axs = plt.subplots(1, len(R.kinds), figsize=(5.2 * len(R.kinds), 4), constrained_layout=True, sharey=True,
                            squeeze=False)
    x = np.arange(R.nlev)
    for ax, k in zip(axs[0], R.kinds):
        bias = np.array([R.band_bias(R.sel(k, b)) for b in range(R.nlev)])      # (levels, bands)
        for j, (name, _, _) in enumerate(BANDS):
            ax.plot(x, bias[:, j], marker="o", color=P.SERIES[j], label=name)
        ax.axhline(0, color=P.INK2, lw=1, ls="--")
        ax.set_xticks(x, lvl_labels(R))
        ax.set_xlabel("Input SNR (dB)")
        ax.set_title(NOISE[k][0] + " segments")
    axs[0, 0].set_ylabel("Output power - clean power (dB)")
    axs[0, -1].legend(fontsize=8)
    P.save(fig, out, "33_band_bias_vs_input_snr")


def p34_spectral_by_noise(R, out):
    f = np.fft.rfftfreq(R.r.T, 1.0 / R.fs)
    keep = (f >= 0.5) & (f <= 40)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    for k in R.kinds:
        s = R.sel(k)
        pc = (np.abs(np.fft.rfft(R.r.clean[s], axis=-1)) ** 2).mean(0)
        pd = (np.abs(np.fft.rfft(R.r.den[s], axis=-1)) ** 2).mean(0)
        pn = (np.abs(np.fft.rfft(R.r.noisy[s], axis=-1)) ** 2).mean(0)
        ax.plot(f[keep], 10 * np.log10((pd[keep] + 1e-20) / (pc[keep] + 1e-20)), color=NOISE[k][1], label=NOISE[k][0] + " output")
        ax.plot(f[keep], 10 * np.log10((pn[keep] + 1e-20) / (pc[keep] + 1e-20)), color=NOISE[k][1], ls=":", lw=1.2, label=NOISE[k][0] + " noisy input")
    ax.axhline(0, color=P.INK2, lw=1, ls="--")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power relative to clean (dB)")
    ax.set_title("Spectral power difference by artifact (all SNR levels pooled)")
    ax.legend(fontsize=8)
    P.save(fig, out, "34_spectral_difference_by_noise")


def p35_vs_paper(R, out):
    keys = [("cc", "CC"), ("rrmse_total", "RRMSE total (t+f+s)/3"), ("snr", "SNR (dB)"), ("psnr", "PSNR (dB)")]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.8), constrained_layout=True)
    w = 0.34
    for ax, (key, title) in zip(axs, keys):
        for i, k in enumerate(R.kinds):
            ours = R.r.seg[key][R.sel(k)].mean()
            paper = PAPER[k][key]
            xs = i
            ax.bar(xs - w / 2, paper, w, color=P.MUTED, label="paper" if i == 0 else None)
            ax.bar(xs + w / 2, ours, w, color=NOISE[k][1], label="ours" if i == 0 else None)
            ax.annotate("%.3g" % paper, (xs - w / 2, paper), ha="center", va="bottom", fontsize=8, color=P.INK2)
            ax.annotate("%.3g" % ours, (xs + w / 2, ours), ha="center", va="bottom", fontsize=8, color=P.INK)
        ax.set_xticks(range(len(R.kinds)), [NOISE[k][0] for k in R.kinds])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, fontsize=9)
    note(fig, "ours = mean over all test segments and all 6 SNR levels (-5..20 dB); paper = Table 5. Test sets are not identical.")
    P.save(fig, out, "35_ours_vs_paper_EOG_EMG")


def p36_pass_rate(R, out):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, (key, thr, hi, title) in zip(axs, [("cc", np.linspace(0.5, 1, 100), True, "segments with CC above threshold"),
                                                 ("rrmse_t", np.linspace(0.05, 1, 100), False, "segments with RRMSE below threshold")]):
        v = R.r.seg[key]
        for k in R.kinds:
            vv = v[R.sel(k)]
            ax.plot(thr, [(vv > t).mean() * 100 if hi else (vv < t).mean() * 100 for t in thr], color=NOISE[k][1], label=NOISE[k][0])
        ax.set_xlabel("threshold")
        ax.set_ylabel("% of segments")
        ax.set_title(key.upper() + ": " + title)
    axs[0].legend()
    P.save(fig, out, "36_pass_rate_by_noise")


def p37_training(R, out):
    h = R.r.hist
    if "val_mse" not in h:
        return False
    ep = np.arange(1, len(h["val_mse"]) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    ax = axs[0]
    for key, lab, c in (("train_mse_eval", "train (inference mode)", P.BLUE), ("val_mse", "validation", P.ORANGE)):
        if key in h:
            ax.plot(ep[: len(h[key])], h[key], color=c, label=lab)
    be = h.get("best_epoch")
    if be:
        ax.axvline(be, color=P.INK2, ls=":", lw=1)
        ax.annotate("best epoch %d" % be, (be, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points", fontsize=8, color=P.INK2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Train vs validation (overfitting check)")
    ax.legend()
    ax = axs[1]
    if "lr" in h:
        ax.plot(ep[: len(h["lr"])], h["lr"], color=P.VIOLET)
        ax.set_yscale("log")
        ax.set_title("Learning rate (cosine warm restarts)")
    ax.set_xlabel("epoch")
    P.save(fig, out, "37_overfitting_and_lr")


def tables(R, out):
    rows = [["noise", "input_snr_db", "n", "cc", "rrmse_t", "rrmse_f", "snr_out", "psnr", "gain_db"] +
            ["bias_" + n.split()[0] for n, _, _ in BANDS]]
    for k in R.kinds:
        for b in range(R.nlev):
            s = R.sel(k, b)
            sg = R.r.seg
            rows.append([NOISE[k][0], "%g" % R.levels[b], int(s.sum())] +
                        ["%.4f" % sg[q][s].mean() for q in ("cc", "rrmse_t", "rrmse_f", "snr", "psnr")] +
                        ["%.3f" % R.gain[s].mean()] + ["%.3f" % v for v in R.band_bias(s)])
        s = R.sel(k)
        rows.append([NOISE[k][0], "ALL", int(s.sum())] + ["%.4f" % R.r.seg[q][s].mean() for q in ("cc", "rrmse_t", "rrmse_f", "snr", "psnr")] +
                    ["%.3f" % R.gain[s].mean()] + ["%.3f" % v for v in R.band_bias(s)])
    with open(os.path.join(out, "phase2_table_by_noise_and_snr.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)


SINGLE = [p30_by_noise, p31_gain, p32_band_bias, p33_band_vs_snr, p34_spectral_by_noise, p35_vs_paper, p36_pass_rate, p37_training]


# ------------------------------------------------------------------ comparison of runs
def compare(Rs, out):
    names = [R.label for R in Rs]
    colors = P.SERIES

    def c40(Rs, out):          # band bias, one panel per noise type
        kinds = sorted(set().union(*[R.kinds for R in Rs]))
        fig, axs = plt.subplots(1, len(kinds), figsize=(5.8 * len(kinds), 4.2), constrained_layout=True, sharey=True, squeeze=False)
        x, w = np.arange(len(BANDS)), 0.8 / len(Rs)
        for ax, k in zip(axs[0], kinds):
            for i, R in enumerate(Rs):
                if k in R.kinds:
                    ax.bar(x + (i - (len(Rs) - 1) / 2) * w, R.band_bias(R.sel(k)), w, color=colors[i], label=R.label)
            ax.axhline(0, color=P.INK2, lw=1)
            ax.set_xticks(x, [n.replace(" ", "\n") for n, _, _ in BANDS])
            ax.set_title(NOISE[k][0])
        axs[0, 0].set_ylabel("Output power - clean power (dB)")
        axs[0, 0].legend(fontsize=8)
        P.save(fig, out, "40_compare_band_bias")

    def c41(Rs, out):          # main metrics, grouped bars
        keys = [("cc", "CC"), ("rrmse_t", "RRMSE"), ("snr", "SNR out (dB)"), ("gain", "SNR gain (dB)")]
        kinds = sorted(set().union(*[R.kinds for R in Rs]))
        fig, axs = plt.subplots(len(kinds), 4, figsize=(12, 3.4 * len(kinds)), constrained_layout=True, squeeze=False)
        w = 0.8 / len(Rs)
        for r_i, k in enumerate(kinds):
            for ax, (key, title) in zip(axs[r_i], keys):
                for i, R in enumerate(Rs):
                    if k in R.kinds:
                        v = R.gain if key == "gain" else R.r.seg[key]
                        val = v[R.sel(k)].mean()
                        ax.bar(i, val, 0.7, color=colors[i], label=R.label if r_i == 0 else None)
                        ax.annotate("%.3g" % val, (i, val), ha="center", va="bottom", fontsize=8)
                ax.set_xticks([])
                ax.set_title("%s  (%s)" % (title, NOISE[k][0]))
                ax.grid(axis="x", visible=False)
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=len(Rs), fontsize=9)
        P.save(fig, out, "41_compare_metrics")

    def c42(Rs, out):          # gain vs input SNR
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True, sharey=True)
        for ax, k in zip(axs, (0, 1)):
            for i, R in enumerate(Rs):
                if k in R.kinds:
                    mu, _ = mean_sd(R.gain, R, k)
                    ax.plot(R.levels, mu, marker="o", color=colors[i], label=R.label)
            ax.axhline(0, color=P.INK2, ls="--", lw=1)
            ax.set_title(NOISE[k][0])
            ax.set_xlabel("Input SNR (dB)")
        axs[0].set_ylabel("SNR gain (dB)")
        axs[0].legend(fontsize=8)
        P.save(fig, out, "42_compare_gain_vs_snr")

    for name, fn in (("40", c40), ("41", c41), ("42", c42)):
        P.plot("compare_" + name, out)(fn)(Rs, out)
    rows = [["run", "noise", "cc", "rrmse_t", "snr_out", "psnr", "gain_db"] + ["bias_" + n.split()[0] for n, _, _ in BANDS]]
    for R in Rs:
        for k in R.kinds:
            s = R.sel(k)
            rows.append([R.label, NOISE[k][0]] + ["%.4f" % R.r.seg[q][s].mean() for q in ("cc", "rrmse_t", "snr", "psnr")] +
                        ["%.3f" % R.gain[s].mean()] + ["%.3f" % v for v in R.band_bias(s)])
    with open(os.path.join(out, "phase2_compare_table.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?")
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--fs", type=float, default=256)
    a = ap.parse_args()
    if a.compare:
        labels = a.labels or [os.path.basename(os.path.dirname(os.path.abspath(p))) for p in a.compare]
        Rs = [R2(p, l, a.fs) for p, l in zip(a.compare, labels)]
        out = a.out or "compare_phase2"
        os.makedirs(out, exist_ok=True)
        compare(Rs, out)
    else:
        if not a.run:
            ap.error("give a run folder or use --compare")
        R = R2(a.run, None, a.fs)
        out = a.out or os.path.join(a.run, "plots_phase2")
        os.makedirs(out, exist_ok=True)
        tables(R, out)
        for fn in SINGLE:
            P.plot(fn.__name__, out)(fn)(R, out)
    print("\nwrote %d plots to %s" % (len(P.DONE), os.path.abspath(out)))
    for n in P.DONE:
        print("  ", n + ".png")
    for n, why in P.SKIPPED:
        print("  skipped %s -> %s" % (n, why))


if __name__ == "__main__":
    main()
