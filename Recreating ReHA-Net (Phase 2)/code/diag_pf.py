"""Is the paper-faithful model scored worse than it really is because of BatchNorm / ReVIN train-vs-eval behaviour?

    python diag_pf.py <results>/pf_MIXED_seed0/1 --data-dir <data>

Loads the run's best weights and measures the plain MSE on the validation set three ways:
  A  eval mode                       (what validation / test numbers use: BatchNorm uses its stored running statistics)
  B  BN batch-statistics mode, dropout OFF   (what training effectively did, minus the dropout noise)
  C  eval mode with every BatchNorm layer's running statistics re-estimated from 3 passes over the TRAINING data
If B or C is much lower than A, the model is better than its reported scores and the BatchNorm statistics are the bug.
If A, B and C are all about equal (~0.1), the model itself underfits and the cause is in the architecture / loss.
Inference only, minutes on a GPU.
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf

from data_reha import load_segments, prepare_reha_data
from model_reha_pf import build_reha_net_pf

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--data-dir", required=True)
a = p.parse_args()

cfg = json.load(open(os.path.join(a.run, "nn_output", "metrics_summary.json")))["run"]
if cfg.get("arch") != "pf":
    raise SystemExit("diag_pf.py is for paper-faithful (arch pf) runs; this run is %s" % cfg.get("arch", "v1"))
model = build_reha_net_pf(512, **cfg["arch_cfg"])
model.load_weights(os.path.join(a.run, "best.weights.h5"))
eeg, eog, emg = load_segments(a.data_dir, cfg["noise_type"])
(ntr, ctr, nv, cv, _, _, _, info) = prepare_reha_data(
    eeg, eog, emg, noise_type=cfg["noise_type"], combin_num=cfg.get("combin_num", 10), train_per=0.8, seed=cfg["seed"],
    snr_convention=cfg["snr_convention"], snr_range=tuple(cfg["snr_range"]), n_levels=6,
    use_bandpass=cfg.get("bandpass", False), independent_eval=cfg.get("independent_eval", False), verbose=False)

rs = np.random.RandomState(0)
perm = rs.permutation(len(nv))[:2048]            # mixed SNR levels in every batch
xv, yv = nv[perm].astype(np.float32), cv[perm].astype(np.float32)
if xv.ndim == 2:
    xv, yv = xv[..., None], yv[..., None]


def mse(training):
    out = []
    for i in range(0, len(xv), 256):
        pred = model(xv[i:i + 256], training=training)
        out.append(float(tf.reduce_mean(tf.square(pred - yv[i:i + 256]))))
    return float(np.mean(out))


A = mse(False)
# B: batch statistics, dropout off
dropouts = [l for l in model._flatten_layers() if isinstance(l, tf.keras.layers.Dropout)]
old = [d.rate for d in dropouts]
for d in dropouts:
    d.rate = 0.0
B = mse(True)
for d, r in zip(dropouts, old):
    d.rate = r
print("dropout layers found: %d" % len(dropouts))

# C: re-estimate BatchNorm running statistics from training data, then eval mode
model.load_weights(os.path.join(a.run, "best.weights.h5"))
bns = [l for l in model._flatten_layers() if isinstance(l, tf.keras.layers.BatchNormalization)]
for l in bns:
    l.momentum = 0.9
tr = np.random.RandomState(1).permutation(len(ntr))[:256 * 40]
xt = ntr[tr].astype(np.float32)
xt = xt[..., None] if xt.ndim == 2 else xt
for d in dropouts:
    d.rate = 0.0
for i in range(0, len(xt), 256):
    model(xt[i:i + 256], training=True)
for d, r in zip(dropouts, old):
    d.rate = r
C = mse(False)

print("\nvalidation MSE of %s (2048 validation segments, mixed SNR levels)" % a.run)
print("  A  eval mode (what the reported scores use)         : %.5f" % A)
print("  B  BatchNorm batch statistics, dropout off          : %.5f" % B)
print("  C  eval mode, BatchNorm statistics re-estimated     : %.5f" % C)
print("  BatchNorm layers: %d" % len(bns))
if min(B, C) < 0.85 * A:
    print("VERDICT: the model is better than its reported score: BatchNorm statistics are the problem (B or C << A).")
else:
    print("VERDICT: A, B and C agree within 15%%: BatchNorm statistics are NOT the cause. The model itself underfits.")
