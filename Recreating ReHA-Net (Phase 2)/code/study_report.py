"""Reads the runs made by run_study.py and writes the answers (with numbers) to study_answers.txt + study_table.csv.

    python study_report.py --result-dir <results>

Whatever has not been run yet is reported as "NOT RUN", never guessed.
Needs numpy + matplotlib (plots_phase2 is reused for the band-power bias).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# the paper's "RRMSE" is the TOTAL (mean of temporal, frequency and spectral RRMSE), not the temporal one
PAPER = {"EOG": dict(cc=0.976, rrmse_total=0.1653, snr=17.06, psnr=27.10),
         "EMG": dict(cc=0.976, rrmse_total=0.1718, snr=16.87, psnr=26.91)}
SENS = [("lr_x2", "learning rate x2 (1e-4)"), ("lr_half", "learning rate /2 (2.5e-5)"), ("drop_0", "dropout 0"),
        ("drop_04", "dropout 0.4"), ("cc_05", "CC-loss weight 0.5"), ("psd_0", "FFT-loss weight 0"),
        ("band_03", "band-loss weight 0.3 (vs 0.1 reference band_s0)")]
DBG = [("dbg_drop0", "dropout 0"), ("dbg_revin_input", "ReVIN at the input only"),
       ("dbg_residual", "residual connections in MSC + attention"), ("dbg_combin10", "10x training data (--combin-num 10)")]
ABL = [("no_freq_att", "remove frequency attention"), ("no_temporal_att", "remove temporal attention"),
       ("no_spatial_att", "remove spatial attention"), ("revin_none", "remove ReVIN")]


def load(res, name):
    p = os.path.join(res, name, "1")
    f = os.path.join(p, "nn_output", "metrics_summary.json")
    if not os.path.exists(f):
        return None
    s = json.load(open(f))
    d = {"summary": s, "path": p}
    d["all"] = s["overall"]
    d["EOG"] = s.get("by_noise_EOG", {}).get("overall")
    d["EMG"] = s.get("by_noise_EMG", {}).get("overall")
    d["best_epoch"] = s["run"].get("best_epoch")
    d["epochs_run"] = s["run"].get("epochs_run")
    try:
        import plots_phase2 as P2
        R = P2.R2(p)
        b = R.band_bias(R.sel())
        d["bias"] = {n.split()[0]: float(v) for (n, _, _), v in zip(P2.BANDS, b)}
        d["gain_by_level"] = [float(R.gain[R.sel(None, i)].mean()) for i in range(R.nlev)]
        d["levels"] = [float(v) for v in R.levels]
    except Exception as e:  # noqa: BLE001
        d["bias"] = None
        d["bias_err"] = "%s: %s" % (type(e).__name__, e)
    return d


def f(x, n=4):
    return "n/a" if x is None else ("%." + str(n) + "f") % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    a = ap.parse_args()
    res = a.result_dir
    out = []
    P = out.append
    runs = {}
    for n in ["base_s0", "base_s1", "base_s2", "band_s0", "band_s1", "band_s2", "cnn_s0"] + [x for x, _ in SENS + ABL + DBG]:
        runs[n] = load(res, n)
    have = [n for n, v in runs.items() if v]
    P("RUNS FOUND: " + (", ".join(have) if have else "none"))
    P("RUNS NOT FOUND: " + (", ".join(n for n, v in runs.items() if not v) or "none"))
    P("")

    # ------------------------------------------------------------------ underfit diagnosis
    P("=" * 72 + "\nDIAGNOSIS: WHY IS THE PAPER-FAITHFUL MODEL UNDERFITTING?  (one change vs base_s0, same seed)\n" + "=" * 72)
    b0 = runs["base_s0"]
    if not b0:
        P("NOT RUN: need base_s0 (copy your finished pf_cheap_seed0 folder to base_s0).")
    else:
        def line(lab, r):
            g = r.get("gain_by_level")
            return "%-42s CC %s  RRMSE-tot %s  SNR %s  gain@20dB %s  best epoch %s" % (
                lab, f(r["all"]["cc"]), f(r["all"]["rrmse_total"]), f(r["all"]["snr"], 2), f(g[-1], 2) if g else "n/a", r["best_epoch"])
        P(line("base_s0 (reference)", b0))
        for n, lab_ in DBG:
            P(line(lab_, runs[n]) if runs[n] else "%-42s NOT RUN" % lab_)
        done = [(runs[n]["all"]["rrmse_total"], lab_) for n, lab_ in DBG if runs[n]]
        if done:
            best = min(done)
            P("ANSWER: the change that lowers RRMSE-total most is '%s' (%.4f vs %.4f for the reference)." % (best[1], best[0], b0["all"]["rrmse_total"]))
    P("")

    # ------------------------------------------------------------------ Q6: seed variation
    base = [runs[n] for n in ("base_s0", "base_s1", "base_s2") if runs[n]]
    P("=" * 72 + "\nQ6  IS THE GAP TO THE PAPER LARGER THAN THE VARIATION ACROSS SEEDS?\n" + "=" * 72)
    if len(base) < 2:
        P("NOT RUN: need base_s0 and base_s1 (tier 1) -- one seed cannot give a variation.")
    else:
        P("Baseline (paper-faithful, no band loss), %d seeds; SD is across seeds (the seed also changes the data split)." % len(base))
        P("%-5s %-8s %-10s %-10s %-10s %-10s %s" % ("noise", "metric", "paper", "ours mean", "seed SD", "gap", "|gap| / SD"))
        for k in ("EOG", "EMG"):
            for m in ("cc", "rrmse_total", "snr", "psnr"):
                vals = np.array([b[k][m] for b in base if b[k]])
                if len(vals) < 2:
                    continue
                mu, sd = vals.mean(), vals.std(ddof=1)
                gap = mu - PAPER[k][m]
                P("%-5s %-8s %-10s %-10s %-10s %-10s %s" % (k, m, f(PAPER[k][m]), f(mu), f(sd), f(gap), f(abs(gap) / sd if sd > 0 else float("inf"), 1)))
        P("Reading: a ratio above 2 means the gap is larger than seed-to-seed variation (it is a real gap, not noise).")
        P("Caveat: the paper's test set is not identical to ours, which seed variation cannot explain.")
    P("")

    # ------------------------------------------------------------------ architecture effect
    P("=" * 72 + "\nARCHITECTURE EFFECT: ReHA-Net vs the Phase 1 Simple CNN, SAME data, SAME seed, SAME loss\n" + "=" * 72)
    if not runs["cnn_s0"] or not runs["base_s0"]:
        P("NOT RUN: need base_s0 and cnn_s0 (tier 1).")
    else:
        P("%-14s %-10s %-10s %-10s %-10s" % ("", "CC", "RRMSE-tot", "SNR (dB)", "params"))
        for n, lab_ in (("cnn_s0", "Simple CNN"), ("base_s0", "ReHA-Net pf")):
            r = runs[n]
            P("%-14s %-10s %-10s %-10s %-10s" % (lab_, f(r["all"]["cc"]), f(r["all"]["rrmse_total"]), f(r["all"]["snr"], 2), "{:,}".format(r["summary"]["run"].get("params", 0))))
        dcc = runs["base_s0"]["all"]["cc"] - runs["cnn_s0"]["all"]["cc"]
        P("ANSWER: ReHA-Net minus Simple CNN: CC %+.4f, RRMSE-total %+.4f (one seed; compare with the seed SD in Q6)." % (
            dcc, runs["base_s0"]["all"]["rrmse_total"] - runs["cnn_s0"]["all"]["rrmse_total"]))
    P("")

    # ------------------------------------------------------------------ Phase 3 hypothesis
    P("=" * 72 + "\nPHASE 3: DOES THE BAND-POWER LOSS REMOVE THE LOW-FREQUENCY UNDER-POWER?\n" + "=" * 72)
    pairs = [(runs["base_s%d" % i], runs["band_s%d" % i]) for i in range(3) if runs["base_s%d" % i] and runs["band_s%d" % i]]
    if not pairs:
        P("NOT RUN: need base_sN and band_sN of the same seed (tier 1).")
    else:
        P("Same-seed pairs (same data, same test set): %d" % len(pairs))
        P("%-22s %-12s %-12s %s" % ("quantity", "baseline", "+band loss", "change (mean over pairs)"))
        def row(label, fn, nd=3):
            b = [fn(p[0]) for p in pairs]
            c = [fn(p[1]) for p in pairs]
            if any(v is None for v in b + c):
                return
            P("%-22s %-12s %-12s %s" % (label, f(np.mean(b), nd), f(np.mean(c), nd), f(np.mean(np.array(c) - np.array(b)), nd)))
        for band in ("delta", "theta", "alpha", "beta", "gamma"):
            row("%s bias (dB)" % band, lambda r, band=band: None if not r["bias"] else r["bias"][band], 2)
        row("CC (all)", lambda r: r["all"]["cc"])
        row("RRMSE-t (all)", lambda r: r["all"]["rrmse_t"])
        row("SNR out (all, dB)", lambda r: r["all"]["snr"], 2)
        if len(pairs) >= 2 and all(p[0]["bias"] and p[1]["bias"] for p in pairs):
            d = np.array([abs(p[1]["bias"]["delta"]) - abs(p[0]["bias"]["delta"]) for p in pairs])
            t = np.array([abs(p[1]["bias"]["theta"]) - abs(p[0]["bias"]["theta"]) for p in pairs])
            P("Change in |delta bias| per seed: %s dB; |theta bias|: %s dB (negative = band loss helps)" % (
                ", ".join("%.2f" % v for v in d), ", ".join("%.2f" % v for v in t)))
            if (d < 0).all() and (t < 0).all():
                P("VERDICT: hypothesis supported in every seed (preliminary, reduced-scale).")
            elif (d >= 0).all() and (t >= 0).all():
                P("VERDICT: hypothesis NOT supported: the band loss did not reduce delta/theta bias in any seed.")
            else:
                P("VERDICT: mixed across seeds: not a reliable effect at this scale.")
        base_bias = [b["bias"] for b in base if b["bias"]]
        if base_bias:
            P("Baseline bias averaged over base seeds: " + ", ".join("%s %.2f dB" % (k, np.mean([b[k] for b in base_bias])) for k in base_bias[0]))
            P("(if delta/theta are within ~0.5 dB of zero, the under-power the hypothesis targets does not exist in ReHA-Net)")
    P("")

    # ------------------------------------------------------------------ Q1 sensitivity
    P("=" * 72 + "\nQ1  WHICH HYPERPARAMETER IS THE RESULT MOST SENSITIVE TO?\n" + "=" * 72)
    ref = runs["base_s0"]
    if not ref:
        P("NOT RUN: need base_s0 and the tier-2 runs.")
    else:
        sd = None
        if len(base) >= 2:
            sd = np.std([b["all"]["rrmse_t"] for b in base], ddof=1)
        rows = []
        for n, label in SENS:
            r = runs[n]
            rr = runs["band_s0"] if n == "band_03" else ref
            if not r or not rr:
                rows.append((None, label, "NOT RUN"))
                continue
            d = r["all"]["rrmse_t"] - rr["all"]["rrmse_t"]
            dcc = r["all"]["cc"] - rr["all"]["cc"]
            rows.append((abs(d), label, "RRMSE-t %+.4f (%+.1f%%)   CC %+.4f   best epoch %s" % (
                d, 100 * d / rr["all"]["rrmse_t"], dcc, r["best_epoch"])))
        P("Reference: base_s0, overall RRMSE-t %s, CC %s.  Seed SD of RRMSE-t: %s" % (
            f(ref["all"]["rrmse_t"]), f(ref["all"]["cc"]), f(sd) if sd is not None else "unknown (needs base_s1)"))
        for _, label, txt in sorted(rows, key=lambda x: (x[0] is None, -(x[0] or 0))):
            P("  %-48s %s" % (label, txt))
        ranked = [r for r in rows if r[0] is not None]
        if ranked:
            top = max(ranked, key=lambda x: x[0])
            P("ANSWER: the most sensitive setting tested is '%s' (largest change in RRMSE-t: %.4f)." % (top[1], top[0]))
            if sd is not None:
                P("        It is a real effect only if the change exceeds 2x the seed SD (= %.4f): %s." % (2 * sd, "YES" if top[0] > 2 * sd else "NO, within seed noise"))
    P("")

    # ------------------------------------------------------------------ Q4 ablations
    P("=" * 72 + "\nQ4  WHAT HAPPENS IF ONE COMPONENT IS REMOVED?\n" + "=" * 72)
    if not ref:
        P("NOT RUN: need base_s0 and the tier-3 runs.")
    else:
        for n, label in ABL + [("drop_0", "remove dropout")]:
            r = runs[n]
            if not r:
                P("  %-34s NOT RUN" % label)
                continue
            d = r["all"]["rrmse_t"] - ref["all"]["rrmse_t"]
            P("  %-34s RRMSE-t %s -> %s (%+.1f%%)   CC %s -> %s   SNR %s -> %s" % (
                label, f(ref["all"]["rrmse_t"]), f(r["all"]["rrmse_t"]), 100 * d / ref["all"]["rrmse_t"],
                f(ref["all"]["cc"]), f(r["all"]["cc"]), f(ref["all"]["snr"], 2), f(r["all"]["snr"], 2)))
        P("Reading: positive RRMSE change = the component was helping. Changes smaller than 2x the seed SD (Q1) are noise.")
    P("")

    # ------------------------------------------------------------------ does the model hurt clean inputs
    P("=" * 72 + "\nDOES THE MODEL HURT NEARLY CLEAN INPUTS?  (SNR gain = output SNR - input SNR, dB)\n" + "=" * 72)
    if not ref or "gain_by_level" not in ref:
        P("NOT AVAILABLE: base_s0 missing or plots_phase2 could not load it (%s)." % (ref or {}).get("bias_err", ""))
    else:
        P("input SNR (dB): " + "  ".join("%6g" % v for v in ref["levels"]))
        P("gain (dB):      " + "  ".join("%6.2f" % v for v in ref["gain_by_level"]))
        neg = [l for l, g in zip(ref["levels"], ref["gain_by_level"]) if g < 0]
        P("ANSWER: " + ("YES, the model makes the signal worse at input SNR %s dB." % ", ".join("%g" % v for v in neg) if neg else "NO, the gain is positive at every tested level."))

    txt = "\n".join(out)
    print(txt)
    open(os.path.join(res, "study_answers.txt"), "w").write(txt + "\n")
    with open(os.path.join(res, "study_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "best_epoch", "epochs_run", "cc", "rrmse_t", "snr", "psnr", "cc_EOG", "rrmse_t_EOG", "cc_EMG", "rrmse_t_EMG",
                    "bias_delta", "bias_theta", "bias_alpha", "bias_beta", "bias_gamma"])
        for n, r in runs.items():
            if not r:
                continue
            b = r["bias"] or {}
            w.writerow([n, r["best_epoch"], r["epochs_run"]] + [f(r["all"][m]) for m in ("cc", "rrmse_t", "snr", "psnr")] +
                       [f(r["EOG"]["cc"]) if r["EOG"] else "", f(r["EOG"]["rrmse_t"]) if r["EOG"] else "",
                        f(r["EMG"]["cc"]) if r["EMG"] else "", f(r["EMG"]["rrmse_t"]) if r["EMG"] else ""] +
                       [f(b.get(k), 3) if b else "" for k in ("delta", "theta", "alpha", "beta", "gamma")])
    print("\nwrote %s and %s" % (os.path.join(res, "study_answers.txt"), os.path.join(res, "study_table.csv")))


if __name__ == "__main__":
    main()
