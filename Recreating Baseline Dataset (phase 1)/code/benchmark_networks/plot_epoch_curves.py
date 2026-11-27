"""Per-epoch curves from the extended logging (needs a run made with phase1_logging/train_method.py).
Usage:  python plot_epoch_curves.py <state.pkl> <out_dir> [n_epochs=20] [snr_lo=-7] [snr_hi=2]
Makes (paper-length ones use the first n_epochs, the rest use every epoch):
 21 PSNR and SNR vs epoch (paper Fig. 8)   22 RRMSE and its variants vs epoch (paper Fig. 10)
 23 train / validation / test loss and CC vs epoch   24 learning curve per SNR level   25 gradient norm + epoch time
"""
import sys, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
lo, hi = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (-7.0, 2.0)
os.makedirs(out, exist_ok=True)
h = pickle.load(open(path, "rb")).get("hist", {})
if "val_psnr" not in h or np.all(np.isnan(np.array(h["val_psnr"], float))):
    sys.exit("This state.pkl has no PSNR/SNR history. Train with phase1_logging/train_method.py first.")
A = lambda k: np.array(h[k], float)
E = len(h["val_mse"]); ep_all = np.arange(1, E + 1); m = min(n, E); ep = np.arange(1, m + 1)
if E < n: print(f"WARNING: only {E} epochs available; paper-length titles will say {E}.")
best = int(np.nanargmin(A("val_mse"))) + 1
def finish(fig, name): fig.tight_layout(); fig.savefig(os.path.join(out, name), dpi=200); plt.close(fig); print("saved", name)

# 21: PSNR and SNR (paper Fig. 8)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ep, A("val_psnr")[:m], "o-", ms=4, label="PSNR"); ax.plot(ep, A("val_snr")[:m], "s-", ms=4, label="SNR")
ax.set_xlabel("Epoch"); ax.set_ylabel("dB"); ax.set_title(f"PSNR and SNR across {m} epochs (validation set)"); ax.grid(alpha=.3); ax.legend()
finish(fig, "21_psnr_snr_vs_epoch.png")

# 22: RRMSE and variants (paper Fig. 10) + Zhang's definitions
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for k, lab in (("val_reha_rrmse", "RRMSE (mean of the three)"), ("val_reha_rrmse_t", "RRMSE temporal"),
               ("val_reha_rrmse_s", "RRMSE scaled"), ("val_reha_rrmse_f", "RRMSE frequency")):
    ax[0].plot(ep, A(k)[:m], "o-", ms=3, label=lab)
ax[0].set_title(f"RRMSE and its variants across {m} epochs (paper definitions)")
for k, lab in (("val_rrmse", "RRMSE temporal (RMS / RMS)"), ("val_rrmse_f", "RRMSE spectral (periodogram)")):
    ax[1].plot(ep, A(k)[:m], "o-", ms=3, label=lab)
ax[1].set_title(f"Benchmark (Zhang) definitions, {m} epochs")
for a in ax: a.set_xlabel("Epoch"); a.set_ylabel("RRMSE"); a.grid(alpha=.3); a.legend(fontsize=8)
finish(fig, "22_rrmse_variants_vs_epoch.png")

# 23: train / val / test, loss and CC
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(ep_all, A("train_mse_eval"), label="Train (eval mode)"); ax[0].plot(ep_all, A("train_mse"), ":", label="Train (training mode, dropout on)")
ax[0].plot(ep_all, A("val_mse"), label="Validation")
if not np.all(np.isnan(A("test_mse"))): ax[0].plot(ep_all, A("test_mse"), "--", label="Test (logged only)")
ax[0].axvline(best, c="gray", lw=.8, ls="--"); ax[0].set_ylabel("Loss"); ax[0].set_yscale("log"); ax[0].set_title(f"Loss per epoch (best validation epoch {best})")
ax[1].plot(ep_all, A("train_cc_eval"), label="Train (eval mode)"); ax[1].plot(ep_all, A("val_cc"), label="Validation")
if not np.all(np.isnan(A("test_cc"))): ax[1].plot(ep_all, A("test_cc"), "--", label="Test (logged only)")
ax[1].set_ylabel("CC"); ax[1].set_title("Correlation per epoch")
for a in ax: a.set_xlabel("Epoch"); a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
finish(fig, "23_train_val_test_vs_epoch.png")

# 24: learning curve per SNR level
cc_s = [r for r in h["val_cc_by_snr"] if len(r)]; rr_s = [r for r in h["val_rrmse_by_snr"] if len(r)]
if len(cc_s) == E and E > 0:
    C, R = np.array(cc_s, float), np.array(rr_s, float); L = C.shape[1]; snr = np.linspace(lo, hi, L); cols = plt.cm.viridis(np.linspace(0, .95, L))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for i in range(L):
        ax[0].plot(ep_all, C[:, i], c=cols[i], label=f"{snr[i]:+.0f} dB"); ax[1].plot(ep_all, R[:, i], c=cols[i])
    ax[0].set_ylabel("Validation CC"); ax[1].set_ylabel("Validation RRMSE (time)"); ax[0].legend(title="nominal SNR", fontsize=7, ncol=2)
    for a in ax: a.set_xlabel("Epoch"); a.grid(alpha=.3)
    fig.suptitle("How each noise level learns over time"); finish(fig, "24_learning_curve_per_snr.png")

# 25: gradient norms and epoch time
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
ax[0].plot(ep_all, A("grads"), label="First layer"); ax[0].plot(ep_all, A("grad_norm_all"), label="All layers")
ax[0].set_yscale("log"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Gradient norm"); ax[0].set_title("Gradient norm per epoch"); ax[0].legend(); ax[0].grid(alpha=.3, which="both")
ax[1].bar(ep_all, A("epoch_time"), color="tab:blue"); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Seconds")
ax[1].set_title(f"Time per epoch (total {np.nansum(A('epoch_time'))/60:.0f} min incl. evaluation)"); ax[1].grid(alpha=.3, axis="y")
finish(fig, "25_gradient_and_epoch_time.png")
