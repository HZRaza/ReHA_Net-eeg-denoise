"""Plot training curves from checkpoint/state.pkl (works mid-run).
Usage:  python plot_training_curves.py path/to/state.pkl [resume_epoch ...]
Example: python plot_training_curves.py state.pkl 16
(resume_epoch = first epoch after a restart; drawn as a dashed line)"""
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "state.pkl"
restarts = [int(a) for a in sys.argv[2:]]

s = pickle.load(open(path, "rb"))
train = np.array(s.get("hist", {}).get("train_mse_eval") or s["train_mse_history"], dtype=float)   # eval-mode training loss if logged
val = np.array(s["val_mse_history"], dtype=float)
grad = np.array(s["mse_grads_history"], dtype=float)
ep = np.arange(1, len(train) + 1)

def mark_restarts(ax):
    for k, r in enumerate(restarts):
        ax.axvline(r - 0.5, ls="--", c="gray", lw=1,
                   label="restart (new random split)" if k == 0 else None)

# 1) Loss vs epochs
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ep, train, "o-", ms=3, label="Train MSE")
ax.plot(ep, val, "s-", ms=3, label="Validation MSE")
mark_restarts(ax)
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (standardised signals)")
ax.set_title("Training and validation loss"); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig("loss_vs_epochs.png", dpi=200); plt.close(fig)

# 2) Gradient norm vs epochs
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ep, grad, "o-", ms=3, c="tab:green")
mark_restarts(ax)
ax.set_xlabel("Epoch"); ax.set_ylabel("Gradient norm (first layer)")
ax.set_title("Gradient norm vs epochs"); ax.grid(alpha=.3)
if restarts: ax.legend()
fig.tight_layout(); fig.savefig("gradient_vs_epochs.png", dpi=200); plt.close(fig)

# 3) Per-epoch improvement in validation loss (diminishing returns)
imp = -np.diff(val)
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(ep[1:], imp, color=["tab:red" if x < 0 else "tab:blue" for x in imp])
ax.axhline(0, c="k", lw=.8)
ax.set_xlabel("Epoch"); ax.set_ylabel("Drop in validation MSE vs previous epoch")
ax.set_title("Diminishing returns: how much each epoch still helps")
ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig("val_improvement_per_epoch.png", dpi=200); plt.close(fig)

# Text summary
print(f"Epochs completed: {len(ep)}")
print(f"Train MSE: first {train[0]:.4f} -> last {train[-1]:.4f}")
print(f"Val   MSE: first {val[0]:.4f} -> last {val[-1]:.4f}  (best {val.min():.4f} at epoch {val.argmin()+1})")
print(f"Gap (val - train) at last epoch: {val[-1]-train[-1]:.4f}")
for frac in (0.9, 0.95, 0.99):
    target = val[0] - frac * (val[0] - val[-1])
    e = int(np.argmax(val <= target)) + 1
    print(f"Epoch where {int(frac*100)}% of the total val-loss drop was reached: {e}")
print("Val drop per epoch, last 5:", np.round(imp[-5:], 5))
