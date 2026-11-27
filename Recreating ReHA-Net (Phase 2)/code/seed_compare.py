"""Seed variation, numbers only (numpy, no TensorFlow).   python seed_compare.py <run1> <run2> [<run3> ...]
Each run = results/<name>/1 .  Prints overall / EOG / EMG metrics for every run, then mean, min, max and range across runs
(with only 2 runs report the RANGE, not a standard deviation), the SNR gain by input level, and the paper's EOG row."""
import json
import os
import sys

import numpy as np

PAPER = {"cc": 0.976, "rrmse_total": 0.1653, "psnr": 27.10, "snr": 17.06}
KEYS = ("cc", "rrmse_t", "rrmse_total", "psnr", "snr")


def load(run):
    return json.load(open(os.path.join(run, "nn_output", "metrics_summary.json")))


def table(title, rows, names):
    print("\n" + title)
    print("%-26s" % "" + "".join("%-13s" % k for k in KEYS))
    for n, r in zip(names, rows):
        print("%-26s" % n + "".join("%-13.4f" % r[k] for k in KEYS))
    if len(rows) > 1:
        for lab, f in (("mean", np.mean), ("min", np.min), ("max", np.max)):
            print("%-26s" % lab + "".join("%-13.4f" % f([r[k] for r in rows]) for k in KEYS))
        print("%-26s" % "range (max-min)" + "".join("%-13.4f" % (max(r[k] for r in rows) - min(r[k] for r in rows)) for k in KEYS))
        m = np.mean([r["rrmse_total"] for r in rows])
        rg = max(r["rrmse_total"] for r in rows) - min(r["rrmse_total"] for r in rows)
        print("  RRMSE-total range is %.1f%% of its mean" % (100 * rg / m))


runs = sys.argv[1:]
if not runs:
    raise SystemExit(__doc__)
S = [load(r) for r in runs]
names = ["%s (seed %s)" % (os.path.basename(os.path.dirname(os.path.abspath(r))), s["run"].get("seed", "?")) for r, s in zip(runs, S)]
print("SNR convention per run:", [s["run"].get("snr_convention") for s in S], "| epochs run:", [s["run"].get("epochs_run") for s in S],
      "| best epoch:", [s["run"].get("best_epoch") for s in S])
table("ALL TEST SEGMENTS (mixed EOG + EMG)", [s["overall"] for s in S], names)
for nz in ("EOG", "EMG"):
    if all(("by_noise_" + nz) in s for s in S):
        table(nz + " only", [s["by_noise_" + nz]["overall"] for s in S], names)
print("\nPAPER (Table 5, EOG): CC %.3f  RRMSE-total %.4f  PSNR %.2f  SNR %.2f   <- compare with the EOG block, not the mixed one"
      % (PAPER["cc"], PAPER["rrmse_total"], PAPER["psnr"], PAPER["snr"]))
print("\nSNR gain by input level (dB; output SNR minus input SNR, standard dB):")
for n, s in zip(names, S):
    per = s.get("per_snr", [])
    if per:
        print("%-26s" % n + "  ".join("%+6.2f" % (p["snr"] - p["snr_input_db"]) for p in per))
if S and S[0].get("per_snr"):
    print("%-26s" % "nominal input level" + "  ".join("%6g" % p["snr_nominal_db"] for p in S[0]["per_snr"]))
