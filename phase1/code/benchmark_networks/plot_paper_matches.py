"""Two plots that mirror ReHA-Net paper figures, made from the finished 50-epoch run.
Usage: python plot_paper_matches.py <nn_output_dir> <state.pkl> <out_dir>"""
import sys, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

nn_dir, state_path, out = sys.argv[1:4]
restarts = [int(a) for a in sys.argv[4:]]      # optional: first epoch after each restart (none for seeded runs)
os.makedirs(out, exist_ok=True)

# 1) Training and validation loss over the first 20 epochs (paper Fig. 7)
s = pickle.load(open(state_path, "rb"))
tr = np.array(s.get("hist", {}).get("train_mse_eval") or s["train_mse_history"], float)[:20]   # eval-mode training loss if logged
va = np.array(s["val_mse_history"], float)[:20]
n = len(va); ep = np.arange(1, n + 1)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ep, tr, "o-", ms=4, label="Training loss")
ax.plot(ep, va, "s-", ms=4, label="Validation loss")
for k, r in enumerate(restarts):
    ax.axvline(r - 0.5, ls="--", c="gray", lw=1, label="restart (new random split)" if k == 0 else None)
ax.set_title(f"Training and validation loss across {n} epochs")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_xticks(ep[::2]); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(out, "8_loss_first20_epochs.png"), dpi=200); plt.close(fig)

# 2) Spectral power difference vs clean, 0-40 Hz (paper Fig. 15)
clean = np.load(os.path.join(nn_dir, "EEG_test.npy")); noisy = np.load(os.path.join(nn_dir, "noiseinput_test.npy"))
den = np.load(os.path.join(nn_dir, "Denoiseoutput_test.npy")).reshape(clean.shape)
f, Pc = welch(clean, fs=256, nperseg=256); _, Pn = welch(noisy, fs=256, nperseg=256); _, Pd = welch(den, fs=256, nperseg=256)
m = f <= 40
db = lambda P: 10 * np.log10(P.mean(0)[m] / Pc.mean(0)[m])
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(f[m], db(Pn), "r--", label="Noisy input")
ax.plot(f[m], db(Pd), "b-", lw=2, label="Simple CNN output")
ax.axhline(0, c="k", lw=.8)
ax.set_title("Spectral power difference (output - clean), 0-40 Hz")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power difference (dB)"); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(out, "9_spectral_difference.png"), dpi=200); plt.close(fig)
print("max |dB| of model output over 0-40 Hz:", np.abs(db(Pd)).max().round(2), "| noisy:", np.abs(db(Pn)).max().round(2))
