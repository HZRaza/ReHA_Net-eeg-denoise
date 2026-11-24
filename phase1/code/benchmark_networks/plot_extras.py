"""Extra Phase-1 plots + tables for the baseline run (paper-comparison, error analysis, classical baseline, data views).
Usage:  python plot_extras.py <nn_output_dir> <state.pkl> <out_dir> [snr_lo snr_hi]
        (snr_lo/snr_hi = nominal SNR range of the test set, default -7 2)
Outputs (numbered after plot_results.py's 1-9):
 10 metrics vs paper (Table 5 / Fig 11)   11 model comparison bars (Fig 5 / Table 4)   12 error CDF
 13 pass-rate curves                      14 error vs amplitude                         15 best / worst segments
 16 SNR x segment error heatmap           17 per-band CC                                18 classical high-pass baseline
 19 data overview                         20 convergence thresholds                     + two CSV tables
"""
import sys, os, pickle, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, sosfiltfilt

nn_dir, state_path, out = sys.argv[1:4]
lo, hi = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (-7.0, 2.0)
os.makedirs(out, exist_ok=True)
FS, N_SNR = 256, 10
SNR = np.linspace(lo, hi, N_SNR)
save = lambda fig, name: (fig.tight_layout(), fig.savefig(os.path.join(out, name), dpi=200), plt.close(fig))

clean = np.load(os.path.join(nn_dir, "EEG_test.npy")).astype(np.float64)
noisy = np.load(os.path.join(nn_dir, "noiseinput_test.npy")).astype(np.float64)
den = np.load(os.path.join(nn_dir, "Denoiseoutput_test.npy")).reshape(clean.shape).astype(np.float64)
PER = len(clean) // N_SNR
blk = np.arange(len(clean)) // PER
t = np.arange(clean.shape[1]) / FS
rms = lambda a: np.sqrt(np.mean(a ** 2, axis=1))


def metrics(est, eps=1e-8):
    err = clean - est; mse = np.mean(err ** 2, 1); sd = clean.std(1)
    a = est - est.mean(1, keepdims=True); b = clean - clean.mean(1, keepdims=True)
    Fc, Fe = np.abs(np.fft.rfft(clean, axis=1)), np.abs(np.fft.rfft(est, axis=1))
    zc = b / (sd[:, None] + eps); ze = a / (est.std(1)[:, None] + eps)
    m = dict(mse=mse,
             psnr=20 * np.log10(np.max(np.abs(clean), 1) / (np.sqrt(mse) + eps)),
             snr=10 * np.log10(np.sum(clean ** 2, 1) / (np.sum(err ** 2, 1) + eps)),
             cc=np.sum(a * b, 1) / (np.sqrt(np.sum(a ** 2, 1) * np.sum(b ** 2, 1)) + eps),
             rrmse_t=np.sqrt(mse) / (sd + eps),
             rrmse_s=np.sqrt(np.mean((zc - ze) ** 2, 1)),
             rrmse_f=np.sqrt(np.mean((Fc - Fe) ** 2, 1)) / (np.sqrt(np.mean(Fc ** 2, 1)) + eps))
    m["rrmse"] = (m["rrmse_t"] + m["rrmse_s"] + m["rrmse_f"]) / 3
    return m

M, B = metrics(den), metrics(noisy)
mean = lambda d, k: float(np.mean(d[k]))

# ---- tables -------------------------------------------------------------------------------------------------
with open(os.path.join(out, "metrics_by_snr_mean_sd.csv"), "w", newline="") as fh:
    w = csv.writer(fh); keys = ["psnr", "snr", "cc", "rrmse", "rrmse_t", "rrmse_s", "rrmse_f", "mse"]
    w.writerow(["nominal_snr_db", "n"] + [f"{k}_{s}" for k in keys for s in ("mean", "sd")])
    for i in range(N_SNR):
        m = blk == i; w.writerow([f"{SNR[i]:+.1f}", int(m.sum())] + [f"{f(M[k][m]):.4f}" for k in keys for f in (np.mean, np.std)])
    m = np.ones(len(clean), bool); w.writerow(["all", len(clean)] + [f"{f(M[k]):.4f}" for k in keys for f in (np.mean, np.std)])

# ---- 10: ours vs paper Table 5 (EOG row) ---------------------------------------------------------------------
paper = dict(psnr=27.10, snr=17.06, cc=0.976, rrmse=0.1653, rrmse_s=0.1714, rrmse_t=0.1785, rrmse_f=0.1461)
with open(os.path.join(out, "table5_vs_paper.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["metric", "ours", "paper_ReHA-Net_EOG"])
    for k in paper: w.writerow([k, f"{mean(M, k):.4f}", paper[k]])
fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
for a, ks, lab in ((ax[0], ["psnr", "snr"], "dB (higher is better)"), (ax[1], ["cc"], "CC (higher is better)"),
                   (ax[2], ["rrmse", "rrmse_s", "rrmse_t", "rrmse_f"], "RRMSE (lower is better)")):
    x = np.arange(len(ks)); o = [mean(M, k) for k in ks]; p = [paper[k] for k in ks]
    b1 = a.bar(x - .2, o, .4, label="Ours (Simple CNN, original data)", color="tab:red"); b2 = a.bar(x + .2, p, .4, label="Paper: ReHA-Net (EOG)", color="tab:blue")
    a.bar_label(b1, fmt="%.3g", fontsize=8); a.bar_label(b2, fmt="%.3g", fontsize=8)
    a.set_xticks(x); a.set_xticklabels([k.upper().replace("_", "-") for k in ks]); a.set_ylabel(lab); a.grid(alpha=.3, axis="y")
ax[0].legend(fontsize=8); fig.suptitle("Ours vs ReHA-Net, EOG (paper Table 5 / Fig. 11). Different data range and model, so read as a gap, not a like-for-like test")
save(fig, "10_metrics_vs_paper.png")

# ---- 11: model comparison bars (paper Table 4, EOG columns) --------------------------------------------------
T4 = {"ReHA-Net": (0.1653, 0.976, 17.060), "MSCGRU": (0.2460, 0.951, 14.691), "EEGIFNet": (0.2150, 0.968, 14.849),
      "EEGDiR": (0.2780, 0.939, 12.811), "GCTNet": (0.2640, 0.947, 13.590), "NovelGAN": (0.3170, 0.942, 10.580),
      "DuoCL": (0.3090, 0.922, 12.252), "EEGDNet": (0.2700, 0.955, 12.224), "DeepSeparator": (0.3780, 0.923, 8.706),
      "NovelCNN": (0.3470, 0.902, 11.443), "Simple CNN\n(Zhang arXiv)": (0.359, 0.916, np.nan),
      "Simple CNN\n(ours)": (mean(M, "rrmse_t"), mean(M, "cc"), mean(M, "snr"))}
names = list(T4); col = ["tab:blue"] + ["lightgray"] * 9 + ["tab:orange", "tab:red"]
fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for i, (a, lab) in enumerate(zip(ax, ["RRMSE (lower is better)", "CC (higher is better)", "SNR dB (higher is better)"])):
    v = [T4[n][i] for n in names]; bars = a.bar(range(len(names)), v, color=col); a.bar_label(bars, fmt="%.3g", fontsize=8)
    a.set_ylabel(lab); a.grid(alpha=.3, axis="y")
    if i == 1: a.set_ylim(0.85, 1.0)
ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
fig.suptitle("EOG artifact removal: paper Table 4 values vs this run (ours: original -7..+2 dB data, single run)")
save(fig, "11_model_comparison_bars.png")

# ---- 12: error CDF ------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.6))
for sel, lab, c in ((np.ones(len(clean), bool), "All segments", "k"), (blk == 0, f"{SNR[0]:+.0f} dB (hardest)", "tab:red"),
                    (blk == N_SNR // 2, f"{SNR[N_SNR//2]:+.0f} dB", "tab:orange"), (blk == N_SNR - 1, f"{SNR[-1]:+.0f} dB (easiest)", "tab:green")):
    v = np.sort(M["rrmse_t"][sel]); ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=lab, c=c)
ax.set_xlabel("Per-segment RRMSE (time)"); ax.set_ylabel("Fraction of segments with error <= x"); ax.set_xlim(0, 1.5)
ax.set_title("Error distribution (CDF)"); ax.grid(alpha=.3); ax.legend()
save(fig, "12_error_cdf.png")

# ---- 13: pass-rate curves -----------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
th = np.linspace(0.5, 0.99, 100)
for d, lab, c in ((M, "Model", "tab:red"), (B, "Noisy input", "gray")):
    ax[0].plot(th, [100 * np.mean(d["cc"] >= x) for x in th], label=lab, c=c)
th2 = np.linspace(0.1, 1.5, 100)
for d, lab, c in ((M, "Model", "tab:red"), (B, "Noisy input", "gray")):
    ax[1].plot(th2, [100 * np.mean(d["rrmse_t"] <= x) for x in th2], label=lab, c=c)
ax[0].set_xlabel("CC threshold"); ax[0].set_ylabel("% of test segments with CC >= threshold")
ax[1].set_xlabel("RRMSE (time) threshold"); ax[1].set_ylabel("% of test segments with RRMSE <= threshold")
for a in ax: a.grid(alpha=.3); a.legend()
fig.suptitle("Pass-rate curves"); save(fig, "13_pass_rate_curves.png")

# ---- 14: error vs amplitude ---------------------------------------------------------------------------------
amp = rms(clean); edges = np.quantile(amp, np.linspace(0, 1, 11)); idx = np.clip(np.digitize(amp, edges[1:-1]), 0, 9)
med = np.array([np.median(M["rrmse_t"][idx == k]) for k in range(10)]); q1 = np.array([np.percentile(M["rrmse_t"][idx == k], 25) for k in range(10)])
q3 = np.array([np.percentile(M["rrmse_t"][idx == k], 75) for k in range(10)]); xc = np.array([amp[idx == k].mean() for k in range(10)])
fig, ax = plt.subplots(figsize=(7, 4.5)); ax.plot(xc, med, "o-", c="tab:red", label="median"); ax.fill_between(xc, q1, q3, alpha=.2, color="tab:red", label="25-75%")
ax.set_xlabel("Clean-segment RMS (standardised units, decile means)"); ax.set_ylabel("RRMSE (time)"); ax.set_title("Error vs signal amplitude")
ax.grid(alpha=.3); ax.legend(); save(fig, "14_error_vs_amplitude.png")

# ---- 15: best / worst segments ------------------------------------------------------------------------------
order = np.argsort(M["cc"]); pick = list(order[-3:][::-1]) + list(order[:3])
fig, axes = plt.subplots(2, 3, figsize=(15, 6.5), sharex=True)
for a, i, kind in zip(axes.ravel(), pick, ["Best"] * 3 + ["Worst"] * 3):
    a.plot(t, noisy[i], c="lightgray", lw=1); a.plot(t, clean[i], c="k", lw=1.3); a.plot(t, den[i], c="tab:red", lw=1.1)
    a.set_title(f"{kind}: SNR {SNR[blk[i]]:+.0f} dB, CC {M['cc'][i]:.2f}", fontsize=10); a.grid(alpha=.3)
axes[0, 0].legend(["Noisy", "Clean", "Model"], fontsize=8); fig.supxlabel("Time (s)"); fig.suptitle("Best and worst test segments by CC")
save(fig, "15_best_worst_segments.png")

# ---- 16: SNR x segment error heatmap ------------------------------------------------------------------------
if len(clean) % N_SNR == 0:
    H = M["rrmse_t"].reshape(N_SNR, PER); H = H[:, np.argsort(H.mean(0))]
    fig, ax = plt.subplots(figsize=(11, 4.2)); im = ax.imshow(H, aspect="auto", cmap="magma", vmax=np.percentile(H, 98), origin="lower")
    ax.set_yticks(range(N_SNR)); ax.set_yticklabels([f"{s:+.0f}" for s in SNR]); ax.set_ylabel("Nominal input SNR (dB)")
    ax.set_xlabel("Test segments (sorted by mean error)"); ax.set_title("RRMSE (time): SNR x segment"); fig.colorbar(im, label="RRMSE")
    save(fig, "16_snr_segment_heatmap.png")

# ---- 17: per-band CC ----------------------------------------------------------------------------------------
bands = {"delta\n1-4 Hz": (1, 4), "theta\n4-8 Hz": (4, 8), "alpha\n8-13 Hz": (8, 13), "beta\n13-30 Hz": (13, 30), "gamma\n30-40 Hz": (30, 40)}
def bcc(x, y):
    a = x - x.mean(1, keepdims=True); b = y - y.mean(1, keepdims=True)
    return float(np.mean(np.sum(a * b, 1) / (np.sqrt(np.sum(a ** 2, 1) * np.sum(b ** 2, 1)) + 1e-12)))
res = {"Noisy input": [], "Model": []}
for lo_f, hi_f in bands.values():
    sos = butter(4, [lo_f, hi_f], btype="band", fs=FS, output="sos"); fc = sosfiltfilt(sos, clean, axis=1)
    res["Noisy input"].append(bcc(sosfiltfilt(sos, noisy, axis=1), fc)); res["Model"].append(bcc(sosfiltfilt(sos, den, axis=1), fc))
fig, ax = plt.subplots(figsize=(8, 4.5)); x = np.arange(len(bands))
for j, (k, c) in enumerate((("Noisy input", "gray"), ("Model", "tab:red"))):
    b = ax.bar(x + (j - .5) * .38, res[k], .38, label=k, color=c); ax.bar_label(b, fmt="%.2f", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(list(bands)); ax.set_ylabel("CC with clean EEG in that band"); ax.set_title("Per-frequency-band correlation")
ax.grid(alpha=.3, axis="y"); ax.legend(); save(fig, "17_per_band_cc.png")

# ---- 18: classical high-pass baseline ------------------------------------------------------------------------
cuts = [0.5, 1, 2, 3, 4, 5, 6, 8]; hp_cc, hp_rr, hp_out = [], [], {}
for c in cuts:
    out_hp = sosfiltfilt(butter(4, c, btype="high", fs=FS, output="sos"), noisy, axis=1); m = metrics(out_hp)
    hp_cc.append(mean(m, "cc")); hp_rr.append(mean(m, "rrmse_t")); hp_out[c] = m
best = cuts[int(np.argmax(hp_cc))]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].plot(cuts, hp_cc, "o-", label="High-pass filter (4th-order Butterworth)"); ax[0].axhline(mean(M, "cc"), c="tab:red", label="Model"); ax[0].axhline(mean(B, "cc"), c="gray", ls="--", label="Noisy input")
ax[0].set_xlabel("Cut-off (Hz)"); ax[0].set_ylabel("Mean CC"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
ax[0].set_title(f"Cut-off sweep (best {best} Hz, picked ON the test set, so it favours the filter)", fontsize=9)
w = .27
for j, (d, lab, c) in enumerate(((B, "Noisy input", "gray"), (hp_out[best], f"High-pass {best} Hz", "tab:blue"), (M, "Model", "tab:red"))):
    ax[1].bar(np.arange(N_SNR) + (j - 1) * w, [d["cc"][blk == i].mean() for i in range(N_SNR)], w, label=lab, color=c)
ax[1].set_xticks(range(N_SNR)); ax[1].set_xticklabels([f"{s:+.0f}" for s in SNR]); ax[1].set_xlabel("Nominal input SNR (dB)"); ax[1].set_ylabel("Mean CC")
ax[1].set_ylim(0, 1); ax[1].grid(alpha=.3, axis="y"); ax[1].legend(fontsize=8); fig.suptitle("Classical baseline: high-pass filter vs the network")
save(fig, "18_classical_baseline.png")

# ---- 19: data overview --------------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 9)); noise = noisy - clean
a = ax[0, 0]; seg = 0
# every row was divided by its own noisy std, so rescale each noisy trace to the clean segment's amplitude before stacking
traces = [noisy[j * PER + seg] * (rms(clean[[seg]])[0] / rms(clean[[j * PER + seg]])[0]) for j in range(N_SNR)]
step = 2.2 * np.percentile(np.abs(np.concatenate(traces)), 99.5)
for j, tr_ in enumerate(traces): a.plot(t, tr_ + step * j, lw=.8, c="gray")
a.plot(t, clean[seg] + step * N_SNR, c="k", lw=1.2); a.set_yticks([step * j for j in range(N_SNR)] + [step * N_SNR]); a.set_yticklabels([f"{s:+.0f} dB" for s in SNR] + ["clean"])
a.set_xlabel("Time (s)"); a.set_title("One segment mixed at each nominal SNR level")
f, Pc = welch(clean, fs=FS, nperseg=256); _, Pn = welch(noise, fs=FS, nperseg=256); _, Py = welch(noisy, fs=FS, nperseg=256)
a = ax[0, 1]; a.semilogy(f, Pc.mean(0), "k", label="Clean EEG"); a.semilogy(f, Pn.mean(0), "tab:blue", label="Added noise (EOG)"); a.semilogy(f, Py.mean(0), "gray", ls="--", label="Noisy input")
a.set_xlim(0, 80); a.set_ylim(1e-7, 2); a.set_xlabel("Frequency (Hz)"); a.set_ylabel("Mean PSD"); a.set_title("Signal vs noise spectra"); a.legend(); a.grid(alpha=.3, which="both")
a = ax[1, 0]; a.hist(clean.ravel(), bins=150, density=True, alpha=.6, label="Clean", range=(-6, 6)); a.hist(noisy.ravel(), bins=150, density=True, alpha=.5, label="Noisy", range=(-6, 6))
a.set_xlabel("Amplitude (standardised)"); a.set_title("Amplitude histograms"); a.legend(); a.grid(alpha=.3)
a = ax[1, 1]; meas = [np.mean(20 * np.log10(rms(clean[blk == i]) / rms(noise[blk == i]))) for i in range(N_SNR)]
a.plot(SNR, meas, "o-", c="tab:red", label="Measured 20*log10(RMS clean / RMS noise)"); a.plot(SNR, SNR, "k--", label="y = x")
a.set_xlabel("Nominal SNR (dB)"); a.set_ylabel("Measured SNR (dB)"); a.set_title("Nominal vs measured SNR"); a.legend(fontsize=8); a.grid(alpha=.3)
save(fig, "19_data_overview.png")

# ---- 20: convergence thresholds (needs state.pkl) -----------------------------------------------------------
s = pickle.load(open(state_path, "rb")); va = np.array(s["val_mse_history"], float)
targets = [x for x in (0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.055, 0.05) if va.min() <= x]
ep_to = [int(np.argmax(va <= x)) + 1 for x in targets]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].plot(np.arange(1, len(va) + 1), va, "o-", ms=3)
for x, e in zip(targets, ep_to): ax[0].axhline(x, ls=":", c="gray", lw=.8); ax[0].plot(e, x, "r*")
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Validation loss"); ax[0].set_title("Validation loss and target levels"); ax[0].grid(alpha=.3)
b = ax[1].bar([f"{x:g}" for x in targets], ep_to, color="tab:blue"); ax[1].bar_label(b); ax[1].set_xlabel("Target validation loss"); ax[1].set_ylabel("First epoch at or below target")
ax[1].set_title("Epochs needed to reach each target"); ax[1].grid(alpha=.3, axis="y"); save(fig, "20_convergence_thresholds.png")
print("done:", len(os.listdir(out)), "files in", out)
