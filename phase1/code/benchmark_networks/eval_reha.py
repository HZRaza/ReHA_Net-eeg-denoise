"""Re-score a trained ReHA-Net run on a test set with a different SNR range / convention. No training.

  python eval_reha.py "<results>\\reha_MIXED_baseline\\1"                      # reproduces the run's own test numbers
  python eval_reha.py "<results>\\reha_MIXED_baseline\\1" --snr-convention repo # same nominal -5..20 dB, but 2x real dB
  python eval_reha.py "<results>\\reha_MIXED_baseline\\1" --snr-low 10 --snr-high 40

The model settings (width, ReVIN, attention) and the seed are read from the run's metrics_summary.json, so
the same train / validation / test segments are used; only the SNR levels of the test blocks change.
Writes nn_output/eval_<tag>.json next to the run's own metrics.
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf

from data_reha import load_segments, prepare_reha_data
from eeg_utils import format_summary, summarize
from model_reha import build_reha_net
from train_method import predict

DATA_DIR = r"D:/People/hadiacode/New folder/EEGdenoiseNet-master/EEGdenoiseNet-master/data/"  # <- change if needed

p = argparse.ArgumentParser()
p.add_argument("run", help="run folder: results/<foldername>/1")
p.add_argument("--data-dir", default=DATA_DIR)
p.add_argument("--snr-low", type=float, default=None, help="default: the run's own range")
p.add_argument("--snr-high", type=float, default=None)
p.add_argument("--n-levels", type=int, default=6)
p.add_argument("--snr-convention", choices=["repo", "standard"], default=None, help="default: the run's own")
args = p.parse_args()

with open(os.path.join(args.run, "nn_output", "metrics_summary.json")) as f:
    cfg = json.load(f)["run"]
lo = cfg["snr_range"][0] if args.snr_low is None else args.snr_low
hi = cfg["snr_range"][1] if args.snr_high is None else args.snr_high
conv = cfg["snr_convention"] if args.snr_convention is None else args.snr_convention
tag = "%s_%g_%g" % (conv, lo, hi)

model = build_reha_net(512, cfg["base"], cfg["blocks"], cfg["revin"], cfg["temporal_att"], cfg["freq_att"])
model.load_weights(os.path.join(args.run, "best.weights.h5"))
print("loaded %s parameters; evaluating with convention '%s', nominal %g..%g dB" % (
    "{:,}".format(model.count_params()), conv, lo, hi))

if cfg.get("model", "ReHA_Net") != "ReHA_Net":
    raise SystemExit("eval_reha.py only re-scores ReHA-Net runs (this run is %s)." % cfg["model"])
eeg, eog, emg = load_segments(args.data_dir, cfg["noise_type"])
(_, _, _, _, noisy, clean, _, info) = prepare_reha_data(
    eeg, eog, emg, noise_type=cfg["noise_type"], combin_num=cfg.get("combin_num", 10), train_per=0.8, seed=cfg["seed"],   # MUST equal the training run's value (10): it fixes which noise segment is paired with each test segment
    snr_convention=conv, snr_range=(lo, hi), n_levels=args.n_levels, use_bandpass=cfg.get("bandpass", False))

den = predict(model, noisy, "ReHA_Net")
levels, nl = info["snr_levels"], info["n_levels"]
out = summarize(den, clean, noisy, fs=256, n_levels=nl, snr_levels=levels)
lab = np.asarray(info["test_noise_label"])
if cfg["noise_type"] == "MIXED":
    for k, name in ((0, "EOG"), (1, "EMG")):
        s = lab == k
        out["by_noise_" + name] = summarize(den[s], clean[s], noisy[s], fs=256, n_levels=nl, snr_levels=levels)
out["eval"] = {"snr_convention": conv, "snr_range": [lo, hi], "n_levels": nl}

print(format_summary(out, conv))
for name in ("EOG", "EMG"):
    s = out.get("by_noise_" + name)
    if s:
        o = s["overall"]
        print("  %s only: CC %.4f  RRMSE-t %.4f  PSNR %.2f  SNR %.2f  RRMSE total %.4f" % (
            name, o["cc"], o["rrmse_t"], o["psnr"], o["snr"], o["rrmse_total"]))
path = os.path.join(args.run, "nn_output", "eval_%s.json" % tag)
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("saved", path)
