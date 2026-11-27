"""All result plots + metrics table for the EEG denoising run.
Usage:
  python plot_results.py <nn_output_dir> <state.pkl> <out_dir> [restart_epoch ...]
Example:
  python plot_results.py results/EMG_.../1/nn_output results/EMG_.../1/checkpoint/state.pkl plots 16 42
"""
import sys, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

nn_dir, state_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
restarts = [int(a) for a in sys.argv[4:]]
os.makedirs(out, exist_ok=True)
FS = 256                      # EEGdenoiseNet EOG/EEG sampling rate (Hz)
N_SNR = 10                    # 10 SNR levels, one equal block each (test set layout)
SNR_DB = np.linspace(-7, 2, N_SNR)

clean = np.load(os.path.join(nn_dir, "EEG_test.npy"))
noisy = np.load(os.path.join(nn_dir, "noiseinput_test.npy"))
den = np.load(os.path.join(nn_dir, "Denoiseoutput_test.npy")).reshape(clean.shape)
PER = len(clean) // N_SNR       # segments per SNR block (340 on the full test set)
snr_idx = np.arange(len(clean)) // PER
t = np.arange(clean.shape[1]) / FS

# ---------------- metrics ----------------
rms = lambda x, axis=-1: np.sqrt(np.mean(x ** 2, axis=axis))
def psd(x):
    f, p = welch(x, fs=FS, nperseg=256, axis=-1)
    return f, p
def metrics(est):
    mse = np.mean((est - clean) ** 2, axis=1)
    rrmse_t = rms(est - clean) / rms(clean)
    f, pe = psd(est); _, pc = psd(clean)
    rrmse_f = rms(pe - pc) / rms(pc)
    a = est - est.mean(1, keepdims=True); b = clean - clean.mean(1, keepdims=True)
    cc = (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    return dict(MSE=mse, RRMSE_t=rrmse_t, RRMSE_f=rrmse_f, CC=cc)
M, B = metrics(den), metrics(noisy)        # model vs "do nothing" baseline
f, P_clean = psd(clean); _, P_noisy = psd(noisy); _, P_den = psd(den)

# ---------------- table ----------------
import csv
rows = []
for name, d in (("Noisy input (no denoising)", B), ("Denoised (model)", M)):
    rows.append([name, "all"] + [f"{d[k].mean():.4f}" for k in ("MSE", "RRMSE_t", "RRMSE_f", "CC")])
for i, s in enumerate(SNR_DB):
    m = snr_idx == i
    rows.append(["Denoised (model)", f"{s:+.0f} dB"] + [f"{M[k][m].mean():.4f}" for k in ("MSE", "RRMSE_t", "RRMSE_f", "CC")])
with open(os.path.join(out, "metrics_table.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["Signal", "SNR", "MSE", "RRMSE_time", "RRMSE_freq", "CC"]); w.writerows(rows)
for r in rows: print(" | ".join(r))

# ---------------- 1. loss curves ----------------
s = pickle.load(open(state_path, "rb"))
tr = np.array(s.get("hist", {}).get("train_mse_eval") or s["train_mse_history"], float)   # eval-mode training loss if logged
va = np.array(s["val_mse_history"], float)
ep = np.arange(1, len(tr) + 1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ep, tr, "o-", ms=3, label="Train MSE"); ax.plot(ep, va, "s-", ms=3, label="Validation MSE")
for k, r in enumerate(restarts):
    ax.axvline(r - 0.5, ls="--", c="gray", lw=1, label="restart (new random split)" if k == 0 else None)
ax.plot(va.argmin() + 1, va.min(), "*", c="red", ms=14, label=f"best val (epoch {va.argmin()+1})")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (standardised signals)"); ax.set_yscale("log")
ax.set_title("Training and validation loss"); ax.grid(alpha=.3, which="both"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(out, "1_loss_vs_epochs.png"), dpi=200); plt.close(fig)

# ---------------- 2. waveforms ----------------
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for ax, i in zip(axes, (0, N_SNR - 1)):
    idx = np.where(snr_idx == i)[0]
    pick = idx[np.argsort(M["CC"][idx])[len(idx) // 2]]       # median-quality sample
    ax.plot(t, noisy[pick], c="lightgray", lw=1, label="Noisy input")
    ax.plot(t, clean[pick], c="k", lw=1.4, label="Clean EEG (target)")
    ax.plot(t, den[pick], c="tab:red", lw=1.2, label="Model output")
    ax.set_title(f"SNR = {SNR_DB[i]:+.0f} dB  (median-quality segment, CC = {M['CC'][pick]:.2f})")
    ax.set_ylabel("Amplitude (standardised)"); ax.grid(alpha=.3)
axes[0].legend(loc="upper right", ncol=3); axes[1].set_xlabel("Time (s)")
fig.tight_layout(); fig.savefig(os.path.join(out, "2_waveform_comparison.png"), dpi=200); plt.close(fig)
np.save(os.path.join(out, "_example_idx.npy"), pick)

# ---------------- 3. PSD ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, sel, title in ((axes[0], np.ones(len(clean), bool), "All test segments"),
                       (axes[1], snr_idx == 0, "Worst noise level (SNR = -7 dB)")):
    ax.semilogy(f, P_noisy[sel].mean(0), c="gray", label="Noisy input")
    ax.semilogy(f, P_clean[sel].mean(0), c="k", lw=2, label="Clean EEG")
    ax.semilogy(f, P_den[sel].mean(0), c="tab:red", ls="--", lw=2, label="Model output")
    ax.set_title(title); ax.set_xlabel("Frequency (Hz)"); ax.set_xlim(0, 80); ax.grid(alpha=.3, which="both")
axes[0].set_ylabel("Mean power spectral density"); axes[0].legend()
fig.tight_layout(); fig.savefig(os.path.join(out, "3_psd.png"), dpi=200); plt.close(fig)

# ---------------- 4. metrics vs SNR ----------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
for ax, key, lab in zip(axes, ("RRMSE_t", "RRMSE_f", "CC"), ("RRMSE (time)", "RRMSE (frequency)", "Correlation coefficient")):
    for d, c, n in ((B, "gray", "Noisy input"), (M, "tab:red", "Model output")):
        mu = np.array([d[key][snr_idx == i].mean() for i in range(N_SNR)])
        sd = np.array([d[key][snr_idx == i].std() for i in range(N_SNR)])
        ax.plot(SNR_DB, mu, "o-", c=c, label=n); ax.fill_between(SNR_DB, mu - sd, mu + sd, color=c, alpha=.15)
    ax.set_xlabel("Input SNR (dB)"); ax.set_ylabel(lab); ax.grid(alpha=.3, which="both")
    if key == "RRMSE_f":
        ax.set_yscale("log"); ax.set_ylabel(lab + " [log scale]")
        ax.set_ylim(0.05, None)
axes[0].legend(); fig.suptitle("Performance vs noise level (mean ± 1 SD across test segments)")
fig.tight_layout(); fig.savefig(os.path.join(out, "4_metrics_vs_snr.png"), dpi=200); plt.close(fig)

# ---------------- 5. box plots ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, key, lab in zip(axes, ("RRMSE_t", "CC"), ("RRMSE (time)", "Correlation coefficient")):
    ax.boxplot([M[key][snr_idx == i] for i in range(N_SNR)], tick_labels=[f"{s:+.0f}" for s in SNR_DB], showfliers=False)
    ax.set_xlabel("Input SNR (dB)"); ax.set_ylabel(lab); ax.grid(alpha=.3, axis="y")
fig.suptitle("Spread of results across test segments (outliers hidden)")
fig.tight_layout(); fig.savefig(os.path.join(out, "5_boxplots.png"), dpi=200); plt.close(fig)

# ---------------- 6. predicted vs true ----------------
rng = np.random.default_rng(0)
sel = rng.choice(len(clean), 200, replace=False)
x, y = clean[sel].ravel(), den[sel].ravel()
r2 = 1 - np.sum((y - x) ** 2) / np.sum((x - x.mean()) ** 2)
fig, ax = plt.subplots(figsize=(6, 6))
hb = ax.hexbin(x, y, gridsize=80, bins="log", cmap="viridis", mincnt=1)
lim = np.percentile(np.abs(x), 99.9); ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="perfect (y = x)")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
ax.set_xlabel("True clean EEG"); ax.set_ylabel("Model output"); ax.set_title(f"Predicted vs true  (R² = {r2:.2f})")
ax.legend(loc="upper left"); fig.colorbar(hb, label="log10(count)")
fig.tight_layout(); fig.savefig(os.path.join(out, "6_predicted_vs_true.png"), dpi=200); plt.close(fig)
print(f"scatter R2 = {r2:.3f}")

# ---------------- 7. residuals ----------------
res = den - clean
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
axes[0].plot(t, res[pick], c="tab:purple", lw=1); axes[0].axhline(0, c="k", lw=.8)
axes[0].set_title(f"Residual (output - clean), example segment at SNR = {SNR_DB[-1]:+.0f} dB")
axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Residual"); axes[0].grid(alpha=.3)
axes[1].hist(res.ravel(), bins=200, color="tab:purple", density=True)
axes[1].set_title(f"All residuals: mean = {res.mean():+.4f}, SD = {res.std():.3f}")
axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Density"); axes[1].set_xlim(-1.5, 1.5); axes[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(out, "7_residuals.png"), dpi=200); plt.close(fig)
print("done")
