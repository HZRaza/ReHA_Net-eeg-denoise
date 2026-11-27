"""Runs the whole study that answers the examiner's questions, one run after another, resumable.

    python run_study.py --data-dir <data> --result-dir <results> --tiers 1        (the minimum, 6 runs)
    python run_study.py --data-dir <data> --result-dir <results> --tiers 1 2 3    (everything, 16 runs)
    python run_study.py ... --list                                                (just print the plan)

A run whose metrics_summary.json already exists is skipped, so after a Colab disconnect you re-run the same line.
Every run uses the cheap protocol (--combin-num 3).  All runs of one tier share that protocol, so differences between
runs of the SAME seed are caused by the one flag that differs.

Tier 0  diagnosis: why is the paper-faithful model underfitting? (dropout, ReVIN depth, residuals, data size)
Tier 1  baseline vs band loss, seeds 0,1,2 (+ Simple CNN) -> answers Q6 (seed variation) and the Phase 3 hypothesis
Tier 2  one-at-a-time hyperparameter changes, seed 0 -> answers Q1 (most sensitive hyperparameter)
Tier 3  remove one component, seed 0                 -> answers Q4 (what happens if a component is removed)
Then:   python study_report.py --result-dir <results>
"""
import argparse
import os
import subprocess
import sys
import time

COMMON_DEFAULT = "--combin-num 3"
PLAN = [
    # tier, folder name, extra args
    # tier 0 = diagnosis of the underfit seen in the first paper-faithful run (one change each vs base_s0)
    (0, "dbg_drop0", ["--seed", "0", "--dropout", "0"]),
    (0, "dbg_revin_input", ["--seed", "0", "--revin", "input"]),
    (0, "dbg_residual", ["--seed", "0", "--msc-residual", "--att-residual"]),
    (0, "dbg_combin10", ["--seed", "0", "--combin-num", "10"]),
    (1, "base_s0", ["--seed", "0"]),
    (1, "band_s0", ["--seed", "0", "--lambda-band", "0.1"]),
    (1, "base_s1", ["--seed", "1"]),
    (1, "band_s1", ["--seed", "1", "--lambda-band", "0.1"]),
    (1, "base_s2", ["--seed", "2"]),
    (1, "band_s2", ["--seed", "2", "--lambda-band", "0.1"]),
    (1, "cnn_s0", ["--seed", "0", "--network", "simple_cnn"]),
    (2, "lr_x2", ["--seed", "0", "--lr", "1e-4"]),
    (2, "lr_half", ["--seed", "0", "--lr", "2.5e-5"]),
    (2, "drop_0", ["--seed", "0", "--dropout", "0"]),
    (2, "drop_04", ["--seed", "0", "--dropout", "0.4"]),
    (2, "cc_05", ["--seed", "0", "--lambda-cc", "0.5"]),
    (2, "psd_0", ["--seed", "0", "--lambda-psd", "0"]),
    (2, "band_03", ["--seed", "0", "--lambda-band", "0.3"]),
    (3, "no_freq_att", ["--seed", "0", "--no-freq-att"]),
    (3, "no_temporal_att", ["--seed", "0", "--no-temporal-att"]),
    (3, "no_spatial_att", ["--seed", "0", "--no-spatial-att"]),
    (3, "revin_none", ["--seed", "0", "--revin", "none"]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=False)
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--tiers", type=int, nargs="+", default=[1])
    ap.add_argument("--common", default=COMMON_DEFAULT, help="args added to EVERY run, e.g. \"--combin-num 1 --epochs 25 --base 24\"")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    todo = [p for p in PLAN if p[0] in a.tiers]
    for tier, name, extra in todo:
        done = os.path.exists(os.path.join(a.result_dir, name, "1", "nn_output", "metrics_summary.json"))
        print("tier %d  %-16s %-8s %s" % (tier, name, "DONE" if done else "to run", " ".join(extra)))
    if a.list:
        return
    if not a.data_dir:
        ap.error("--data-dir is required to run")
    log = os.path.join(a.result_dir, "study_log.txt")
    os.makedirs(a.result_dir, exist_ok=True)
    for tier, name, extra in todo:
        if os.path.exists(os.path.join(a.result_dir, name, "1", "nn_output", "metrics_summary.json")):
            continue
        cmd = [sys.executable, os.path.join(here, "main_reha.py"), "--foldername", name, "--data-dir", a.data_dir,
               "--result-dir", a.result_dir] + a.common.split() + extra
        print("\n" + "#" * 70 + "\n# RUN %s\n# %s\n" % (name, " ".join(cmd)) + "#" * 70, flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=here)
        line = "%s  %-16s exit=%d  %.1f min" % (time.strftime("%Y-%m-%d %H:%M:%S"), name, rc, (time.time() - t0) / 60)
        print(line, flush=True)
        with open(log, "a") as f:
            f.write(line + "\n")
        if rc != 0:
            print("RUN FAILED: %s. Copy the error above and send it. Continuing with the next run." % name)
    print("\nAll requested runs finished. Now: python study_report.py --result-dir %s" % a.result_dir)


if __name__ == "__main__":
    main()
