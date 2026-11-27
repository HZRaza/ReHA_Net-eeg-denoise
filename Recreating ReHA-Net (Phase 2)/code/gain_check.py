"""How much of the error is just a wrong output AMPLITUDE?  No training, seconds on a laptop, numpy only.

    python gain_check.py <results>/pf_MIXED_seed0/1
    python gain_check.py <results>/run2/1            (works on any run folder with nn_output/*.npy)

For every SNR level it finds the single scalar g that best maps the model output onto the clean EEG (least squares,
g = <clean, out> / <out, out> over all segments of that level), then re-scores the output multiplied by g.
It also reports ONE global g (fitted on the whole test set) and the per-segment optimal g as an upper bound.
If CC barely changes but RRMSE drops a lot, the network's shape is fine and its amplitude calibration is the problem.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eeg_utils import summarize  # noqa: E402

run = sys.argv[1]
d = os.path.join(run, "nn_output")
den = np.load(os.path.join(d, "Denoiseoutput_test.npy")).astype(np.float64)
cln = np.load(os.path.join(d, "EEG_test.npy")).astype(np.float64)
nsy = np.load(os.path.join(d, "noiseinput_test.npy")).astype(np.float64)
n = len(cln)
den, cln, nsy = den.reshape(n, -1), cln.reshape(n, -1), nsy.reshape(n, -1)
z = np.load(os.path.join(run, "split_info.npz"), allow_pickle=True)
levels = [float(v) for v in z["snr_levels"]]
nl = len(levels)
lab = np.asarray(z["test_noise_label"]).astype(int)
m = n // nl


def fit(x, y):
    return float((x * y).sum() / (x * x).sum())


def score(o, tag):
    s = summarize(o, cln, nsy, fs=256, n_levels=nl, snr_levels=levels)
    out = {"tag": tag, "all": s["overall"], "per": s["per_snr"]}
    for k, name in ((0, "EOG"), (1, "EMG")):
        sel = lab == k
        if sel.any():
            out[name] = summarize(o[sel], cln[sel], nsy[sel], fs=256, n_levels=nl, snr_levels=levels)["overall"]
    return out


g_global = fit(den, cln)
g_level = np.ones(n)
for i in range(nl):
    sl = slice(i * m, (i + 1) * m)
    g_level[sl] = fit(den[sl], cln[sl])
g_seg = (den * cln).sum(1) / (den * den).sum(1)

variants = [("as trained", den),
            ("one global gain g=%.3f" % g_global, den * g_global),
            ("one gain per SNR level", den * g_level[:, None]),
            ("per-segment optimal gain (upper bound)", den * g_seg[:, None])]
res = [score(o, t) for t, o in variants]

print("run:", run, "| %d test segments, SNR levels %s" % (n, levels))
print("\nbest gain per SNR level (1.00 = output amplitude already right; >1 means the output is TOO SMALL, <1 TOO BIG):")
print("   " + "  ".join("%g dB: %.3f" % (lv, g_level[i * m]) for i, lv in enumerate(levels)))
print("   mean per-segment optimal gain: %.3f   (global: %.3f)" % (g_seg.mean(), g_global))
print("\n%-42s %-8s %-12s %-8s %-8s | EOG: CC / RRMSE-tot | EMG: CC / RRMSE-tot" % ("", "CC", "RRMSE-total", "SNR", "RRMSE-t"))
for r in res:
    a = r["all"]
    e, g = r.get("EOG"), r.get("EMG")
    print("%-42s %-8.4f %-12.4f %-8.2f %-8.4f | %s | %s" % (
        r["tag"], a["cc"], a["rrmse_total"], a["snr"], a["rrmse_t"],
        "%.4f / %.4f" % (e["cc"], e["rrmse_total"]) if e else "-", "%.4f / %.4f" % (g["cc"], g["rrmse_total"]) if g else "-"))
print("\nSNR gain (output SNR - input SNR, dB) by input level:")
for r in res:
    print("%-42s " % r["tag"] + "  ".join("%6.2f" % (p["snr"] - p["snr_input_db"]) for p in r["per"]))
print("  input SNR (dB):                           " + "  ".join("%6g" % v for v in levels))
base, fixed = res[0]["all"]["rrmse_total"], res[1]["all"]["rrmse_total"]
print("\nVERDICT: one global gain changes RRMSE-total by %+.1f%% (%.4f -> %.4f) and CC by %+.4f." % (
    100 * (fixed - base) / base, base, fixed, res[1]["all"]["cc"] - res[0]["all"]["cc"]))
json.dump({"g_global": g_global, "g_per_level": [float(g_level[i * m]) for i in range(nl)], "variants": res},
          open(os.path.join(d, "gain_check.json"), "w"), indent=1)
print("saved", os.path.join(d, "gain_check.json"))
