"""reha_all.py -- ONE file to run everything for the ReHA-Net project.  Edit CONFIG below, or override from the command line.

    python reha_all.py --data-dir <data> --result-dir <results> --name base_s0 --preset cheap
                                           train -> SNR re-scoring -> plots -> pack a zip you can send

    python reha_all.py ... --dry-run       print every command, run nothing   (do this first)
    python reha_all.py ... --stages train                  only some stages: train rescore plots pack
    python reha_all.py ... --set dropout=0.3 lr=1e-4       override any CONFIG value for this run
    python reha_all.py ... --set lambda_band=0.1           Phase 3: the band-power loss
    python reha_all.py ... --set seed=1 --name base_s1     another seed
    python reha_all.py ... --study 1                       the whole study (tier 1/2/3), then the answers file

What each stage does
  train    main_reha.py  : paper-faithful ReHA-Net (MSC blocks with dropout, hybrid attention, ReVIN), random seed,
                           cosine-restart LR, composite loss; checkpoints every epoch, so re-running RESUMES
  rescore  eval_reha.py  : the trained weights re-scored with other SNR settings (RESCORE list below), inference only
  plots    plot_results.py + plots_phase2.py : the 27 generic plots, then EOG/EMG, SNR gain, band-power bias, vs paper
  pack     one zip (metrics, tables, plots, logs; no big arrays or weights) = what to send me
Every training run writes a log with start/end date and time (needed for the slide screenshot) to <run>/logs/.

Presets (the protocol, not the model, except 'fast' which also makes the network narrower -- use it only to iterate):
  baseline  RESCRIPTED model (arch v2 = v1 + residual path), repo SNR convention, cosine LR, combin 10, 40 epochs, no early stopping  <- default
  run2      run 2's model and protocol (v1, standard dB, cosine restarts) for a like-for-like repeat
  fast      combin 1, 25 epochs, patience 6, base 24   quickest first number; NOT comparable to the baseline
  cheap     combin 3, 40 epochs, no early stopping     3x fewer steps per epoch
  full      combin 10, 40 epochs, no early stopping    (same as baseline; keeps whichever arch/base you set)
"""
import argparse
import ast
import os
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

# =====================================================================================================================
# CONFIG: every value that can matter.  Change here, or with --set key=value on the command line.
# =====================================================================================================================
CONFIG = dict(
    # ---- reproducibility / data
    seed=0,                      # weight init, shuffling AND the train/val/test split of EEG segments
    noise_type="MIXED",          # MIXED (50% EOG / 50% EMG) | EOG | EMG
    combin_num=10,               # noise re-pairings of each training segment (10 = paper-like / run 2, 3 = cheap, 1 = fastest)
    snr_convention="repo",       # repo = EEGdenoiseNet 10*log10 (real dB = 2x nominal; evidence says the paper used it) | standard = 20*log10(rms ratio) | repo = EEGdenoiseNet's 10*log10 (real dB = 2x nominal)
    snr_low=-5.0, snr_high=20.0, n_levels=6,
    fresh_noise=False,           # True = NEW noise pairings / segments / SNRs every epoch (attacks overfitting); same size per epoch
    legacy_eval=False,           # True = reproduce run 2's exact validation/test noise (otherwise independent of combin_num)
    bandpass=False,              # 0.5-40 Hz band-pass of the inputs (paper mentions it; off by default)
    # ---- training
    epochs=40, patience=0, batch_size=32,   # patience 0 = NO early stopping: always all epochs (best-val weights still used for testing)
    lr=5e-5, weight_decay=1e-4, clipnorm=1.0, lr_schedule="cosine", optimizer="adamw",
    # ---- loss:  MSE + lambda_cc*(1-CC) + lambda_psd*FFT-magnitude MAE + lambda_band*per-band log-power (Phase 3)
    lambda_cc=0.05, lambda_psd=0.05, lambda_band=0.0,
    # ---- model (paper-faithful rebuild)
    network="reha",              # reha | simple_cnn (Phase 1 network on the same data)
    arch="v2",                   # v2 = RESCRIPTED BASELINE (v1 + residual path) | v1 = run 2 model | pf = paper-faithful rebuild (underfit in its first run)
    base=36,                     # width of the first stage; v1: 36 = 2.17 M parameters | pf: 46 = 2.15 M
    dropout=0.1,                 # v2: SpatialDropout in the bottleneck | pf: dropout in MSC blocks and attention fusion
    msc_dilation="1,1,1",        # dilation of the k=3,5,7 branches (the paper gives no rates)
    att_reduction=8,             # reduction ratio of the temporal (squeeze-excite) attention
    revin="norm",                # norm = ReVIN at input + after every MSC | input | none | reverse
    msc_residual=False, att_residual=False,
    no_residual=False,           # v2 ablation: remove the global residual path
    no_temporal_att=False, no_spatial_att=False, no_freq_att=False,     # ablation switches
)

PRESETS = {
    "baseline": dict(arch="v2", base=36, combin_num=10, epochs=40, patience=0),   # rescripted baseline, all 40 epochs (the default)
    "run2": dict(arch="v1", snr_convention="standard", lr_schedule="cosine_restarts", dropout=0.2, base=36, combin_num=10, epochs=40, patience=0),   # run 2's model/protocol for a like-for-like repeat
    "fast": dict(combin_num=1, epochs=25, patience=6, base=24),
    "fresh": dict(fresh_noise=True),   # baseline v2 + fresh noise every epoch (Phase 2b)
    "cheap": dict(combin_num=3, epochs=40, patience=0),
    "full": dict(combin_num=10, epochs=40, patience=0),
}

# SNR re-scoring: (label, extra args for eval_reha.py).  Add your own lines.
RESCORE = [
    ("default", []),
    ("standard_convention", ["--snr-convention", "standard"]),
    ("low_snr_-10_0", ["--snr-low", "-10", "--snr-high", "0"]),
    ("high_10_40_standard", ["--snr-convention", "standard", "--snr-low", "10", "--snr-high", "40"]),   # real 10..40 dB
]

FLAG = {  # CONFIG key -> main_reha.py flag
    "seed": "--seed", "noise_type": "--noise-type", "combin_num": "--combin-num", "snr_convention": "--snr-convention",
    "snr_low": "--snr-low", "snr_high": "--snr-high", "n_levels": "--n-levels", "epochs": "--epochs",
    "patience": "--patience", "batch_size": "--batch-size", "lr": "--lr", "weight_decay": "--weight-decay",
    "clipnorm": "--clipnorm", "lr_schedule": "--lr-schedule", "optimizer": "--optimizer", "lambda_cc": "--lambda-cc",
    "lambda_psd": "--lambda-psd", "lambda_band": "--lambda-band", "network": "--network", "arch": "--arch",
    "base": "--base", "dropout": "--dropout", "msc_dilation": "--msc-dilation", "att_reduction": "--att-reduction",
    "revin": "--revin",
}
SWITCH = {"fresh_noise": "--fresh-noise", "no_residual": "--no-residual", "legacy_eval": "--legacy-eval", "bandpass": "--bandpass", "msc_residual": "--msc-residual", "att_residual": "--att-residual",
          "no_temporal_att": "--no-temporal-att", "no_spatial_att": "--no-spatial-att", "no_freq_att": "--no-freq-att"}


def parse_set(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit("--set needs key=value, got %r" % it)
        k, v = it.split("=", 1)
        if k not in CONFIG:
            raise SystemExit("unknown CONFIG key %r. Valid keys: %s" % (k, ", ".join(sorted(CONFIG))))
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
        if k == "msc_dilation" and not isinstance(out[k], str):
            out[k] = ",".join(str(x) for x in out[k])        # --set msc_dilation=1,2,3 -> "1,2,3"
    return out


def train_cmd(cfg, name, a):
    cmd = [sys.executable, os.path.join(HERE, "main_reha.py"), "--foldername", name, "--data-dir", a.data_dir,
           "--result-dir", a.result_dir]
    for k, flag in FLAG.items():
        cmd += [flag, str(cfg[k])]
    for k, flag in SWITCH.items():
        if cfg[k]:
            cmd.append(flag)
    return cmd


def run(cmd, dry, logfile=None, cwd=HERE):
    print("\n$ " + " ".join(cmd), flush=True)
    if dry:
        return 0
    if logfile is None:
        return subprocess.call(cmd, cwd=cwd)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    with open(logfile, "a", encoding="utf-8", errors="replace") as lf:
        lf.write("\n===== START %s =====\n$ %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(cmd)))
        lf.flush()
        p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        buf = b""
        while True:
            ch = p.stdout.read(1)
            if not ch:
                break
            sys.stdout.buffer.write(ch)
            sys.stdout.flush()
            buf += ch
            if ch in (b"\n", b"\r"):
                lf.write(buf.decode("utf-8", "replace"))
                lf.flush()
                buf = b""
        lf.write(buf.decode("utf-8", "replace"))
        rc = p.wait()
        lf.write("\n===== END %s  exit=%d =====\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), rc))
    return rc


def pack(run_dir, name, out_dir):
    keep_ext = (".json", ".csv", ".png", ".txt", ".log", ".pkl", ".npz")
    zpath = os.path.join(out_dir, "share_%s.zip" % name)
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(run_dir):
            for f in files:
                if f.lower().endswith(keep_ext) and "ckpt" not in f:
                    full = os.path.join(root, f)
                    z.write(full, os.path.join(name, os.path.relpath(full, run_dir)))
                    n += 1
    print("\npacked %d files into %s  <- send me this zip" % (n, zpath))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--name", default="run")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--set", nargs="*", metavar="key=value")
    ap.add_argument("--stages", nargs="+", default=["train", "rescore", "plots", "pack"],
                    choices=["train", "rescore", "plots", "pack"])
    ap.add_argument("--study", type=int, nargs="+", metavar="TIER", help="run the study tiers (1 2 3) instead of one run")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.result_dir = a.result_dir.rstrip("/\\")

    cfg = dict(CONFIG)
    if a.preset:
        cfg.update(PRESETS[a.preset])
    cfg.update(parse_set(a.set))
    print("reha_all.py | %s | name=%s" % (time.strftime("%Y-%m-%d %H:%M:%S"), a.name))
    print("config: " + ", ".join("%s=%s" % (k, cfg[k]) for k in sorted(cfg) if cfg[k] != CONFIG[k] or k in ("seed", "combin_num", "base")))

    if a.study:
        common = "--combin-num %d --epochs %d --patience %d --base %d" % (cfg["combin_num"], cfg["epochs"], cfg["patience"], cfg["base"])
        rc = run([sys.executable, os.path.join(HERE, "run_study.py"), "--data-dir", a.data_dir, "--result-dir", a.result_dir,
                  "--tiers"] + [str(t) for t in a.study] + ["--common", common], a.dry_run)
        run([sys.executable, os.path.join(HERE, "study_report.py"), "--result-dir", a.result_dir], a.dry_run)
        return

    run_dir = os.path.join(a.result_dir, a.name, "1")
    if "train" in a.stages:
        rc = run(train_cmd(cfg, a.name, a), a.dry_run, os.path.join(run_dir, "logs", "train.log"))
        if rc != 0:
            print("\nTRAINING FAILED (exit %d). Copy the error above and send it to me. Stopping." % rc)
            sys.exit(rc)
    if "rescore" in a.stages:
        if cfg["network"] != "reha":
            print("\n(re-scoring skipped: eval_reha.py handles ReHA-Net runs only)")
        else:
            for label, extra in RESCORE:
                run([sys.executable, os.path.join(HERE, "eval_reha.py"), run_dir, "--data-dir", a.data_dir] + extra,
                    a.dry_run, os.path.join(run_dir, "logs", "rescore_%s.log" % label))
    if "plots" in a.stages:
        run([sys.executable, os.path.join(HERE, "plot_results.py"), run_dir, "--snr-convention", cfg["snr_convention"]], a.dry_run,
            os.path.join(run_dir, "logs", "plots_generic.log"))
        run([sys.executable, os.path.join(HERE, "plots_phase2.py"), run_dir], a.dry_run,
            os.path.join(run_dir, "logs", "plots_phase2.log"))
    if "pack" in a.stages and not a.dry_run:
        pack(run_dir, a.name, a.result_dir)
    print("\nfinished %s" % time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
