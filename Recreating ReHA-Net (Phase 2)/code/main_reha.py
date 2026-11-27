import argparse
import json
import os

import numpy as np
import tensorflow as tf

from data_reha import load_segments, prepare_reha_data, save_split_info
from eeg_utils import format_summary, summarize
from model_reha import build_reha_net, make_loss
from model_reha_pf import build_reha_net_pf
from model_reha_v2 import build_reha_net_v2
from train_method import predict, train

# Phase 2: ReHA-Net recreation (Keras).  Edit the two paths once, then e.g.
#
#   python main_reha.py --summary                 # build the model, print layers + parameter count, exit
#   python main_reha.py --quick                   # 2-epoch smoke test on a small subset
#   python main_reha.py                           # full run: EOG+EMG, SNR -5..20 dB, 40 epochs
#   python main_reha.py --noise-type EOG --foldername reha_EOG      # EOG-only (compare with the paper's Table 5)
#
# Ctrl+C is safe: run the same command again to resume.

DATA_DIR = r"D:/People/hadiacode/New folder/EEGdenoiseNet-master/EEGdenoiseNet-master/data/"          # <- change if needed
RESULT_DIR = r"D:/People/hadiacode/New folder/EEGdenoiseNet-master/EEGdenoiseNet-master/results"     # <- change if needed

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true")
p.add_argument("--summary", action="store_true", help="print the model summary and exit (no data needed)")
p.add_argument("--foldername", default=None, help="default: reha_MIXED_baseline, or simplecnn_<noise>_enhanced with --network simple_cnn")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--patience", type=int, default=8)
p.add_argument("--noise-type", default="MIXED", choices=["EOG", "EMG", "MIXED"])
p.add_argument("--snr-convention", default="standard", choices=["repo", "standard"])
p.add_argument("--snr-low", type=float, default=-5.0)
p.add_argument("--snr-high", type=float, default=20.0)
p.add_argument("--n-levels", type=int, default=6, help="number of validation/test SNR blocks")
p.add_argument("--bandpass", action="store_true", help="0.5-40 Hz band-pass on EEG and noise before mixing")
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--weight-decay", type=float, default=1e-4)
p.add_argument("--clipnorm", type=float, default=1.0, help="global gradient-norm clip, 0 = off")
p.add_argument("--lr-schedule", default="cosine_restarts", choices=["constant", "step", "exp", "cosine", "cosine_restarts"])
p.add_argument("--lambda-cc", type=float, default=0.05)
p.add_argument("--lambda-psd", type=float, default=0.05)
p.add_argument("--arch", default="pf", choices=["pf", "v1", "v2"],
               help="pf = paper-faithful rebuild (default), v1 = my first recreation (run 2), v2 = rescripted baseline (v1 + residual path)")
p.add_argument("--no-residual", action="store_true", help="v2: drop the global residual path (ablation)")
p.add_argument("--base", type=int, default=None, help="width of the first stage (pf: 46 = ~2.15 M params; v1: 36)")
p.add_argument("--blocks", type=int, default=2, help="MSC blocks per stage")
p.add_argument("--revin", default="norm", choices=["none", "norm", "input", "reverse"], help="pf: norm = input + after every MSC (paper), input = input only")
p.add_argument("--dropout", type=float, default=0.2, help="pf: dropout in MSC blocks and attention fusion")
p.add_argument("--msc-dilation", default="1,1,1", help="pf: dilation of the k=3,5,7 branches, e.g. 1,2,3")
p.add_argument("--att-reduction", type=int, default=8, help="pf: reduction ratio of the temporal (SE) attention")
p.add_argument("--msc-residual", action="store_true", help="pf: add a residual connection around each MSC block")
p.add_argument("--att-residual", action="store_true", help="pf: add a residual connection around each attention module")
p.add_argument("--no-spatial-att", action="store_true", help="pf: drop the spatial branch")
p.add_argument("--combin-num", type=int, default=10, help="training re-pairings of EEG with noise (10 = full, 3 = cheap)")
p.add_argument("--fresh-noise", action="store_true", help="Phase 2b: draw NEW EEG<->noise pairings, noise segments and SNRs every epoch (same size as the fixed set) to attack overfitting")
p.add_argument("--legacy-eval", action="store_true", help="use the first runs' validation/test noise stream (reproduces run 2's exact test set; it changes with --combin-num)")
p.add_argument("--lambda-band", type=float, default=0.0, help="Phase 3: weight of the per-band log-power loss (0 = off)")
p.add_argument("--network", default="reha", choices=["reha", "simple_cnn"], help="simple_cnn = the Phase 1 baseline network, trained on the same enhanced data")
p.add_argument("--optimizer", default="adamw", choices=["adamw", "rmsprop"])
p.add_argument("--no-temporal-att", action="store_true")
p.add_argument("--no-freq-att", action="store_true")
p.add_argument("--data-dir", default=DATA_DIR)
p.add_argument("--result-dir", default=RESULT_DIR)
args = p.parse_args()
if args.foldername is None:
    args.foldername = "reha_MIXED_baseline" if args.network == "reha" else "simplecnn_%s_enhanced" % args.noise_type
net_name = "ReHA_Net" if args.network == "reha" else "Simple_CNN"

tf.keras.utils.set_random_seed(args.seed)
fs, datanum = 256, 512

if args.network == "simple_cnn":
    from Network_structure import simple_CNN   # the Phase 1 network
    arch_cfg = None
    model = simple_CNN(datanum)
elif args.arch == "pf":
    arch_cfg = dict(base=args.base or 46, dropout=args.dropout,
                    dilations=[int(v) for v in args.msc_dilation.split(",")], reduction=args.att_reduction,
                    revin=args.revin, temporal_att=not args.no_temporal_att, spatial_att=not args.no_spatial_att,
                    freq_att=not args.no_freq_att, msc_residual=args.msc_residual, att_residual=args.att_residual)
    model = build_reha_net_pf(datanum, **arch_cfg)
elif args.arch == "v2":
    arch_cfg = dict(base=args.base or 36, blocks=args.blocks, dropout=args.dropout, revin=args.revin,
                    temporal_att=not args.no_temporal_att, freq_att=not args.no_freq_att, residual=not args.no_residual)
    model = build_reha_net_v2(datanum, **arch_cfg)
else:
    arch_cfg = None
    model = build_reha_net(datanum, args.base or 36, args.blocks, args.revin,
                           not args.no_temporal_att, not args.no_freq_att)
if args.summary:
    model.summary()
    print("\nTrainable parameters: {:,}".format(model.count_params()))
    raise SystemExit(0)
print("%s parameters: {:,}".format(model.count_params()) % net_name)

epochs, foldername = args.epochs, args.foldername
if args.quick:
    epochs, foldername = 2, foldername + "_quick"
result_location = args.result_dir.replace("\\", "/").rstrip("/")
train_num = "1"

# ---------------------------------------------------------------- data
EEG_all, EOG_all, EMG_all = load_segments(args.data_dir, args.noise_type)
(noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, _, info) = prepare_reha_data(
    EEG_all, EOG_all, EMG_all, noise_type=args.noise_type, combin_num=args.combin_num, train_per=0.8, seed=args.seed,
    snr_convention=args.snr_convention, snr_range=(args.snr_low, args.snr_high), n_levels=args.n_levels,
    use_bandpass=args.bandpass, fs=fs, independent_eval=not args.legacy_eval)

if args.quick:  # keep every SNR level in val/test, just fewer segments
    noiseEEG_train, EEG_train = noiseEEG_train[:1000], EEG_train[:1000]
    sel_v, sel_t = np.arange(len(EEG_val))[::10], np.arange(len(EEG_test))[::10]
    noiseEEG_val, EEG_val = noiseEEG_val[sel_v], EEG_val[sel_v]
    noiseEEG_test, EEG_test = noiseEEG_test[sel_t], EEG_test[sel_t]
    info["test_noise_label"] = info["test_noise_label"][sel_t]
    info["n_levels"] = 1  # a stride-10 subset no longer keeps equal SNR blocks

run_dir = os.path.join(result_location, foldername, train_num)
os.makedirs(run_dir, exist_ok=True)
save_split_info(info, os.path.join(run_dir, "split_info.npz"))

# ---------------------------------------------------------------- optimizer / loss
opt_kw = {"learning_rate": args.lr, "weight_decay": args.weight_decay}
if args.clipnorm > 0:
    opt_kw["global_clipnorm"] = args.clipnorm
if args.optimizer == "adamw":
    optimizer = tf.keras.optimizers.AdamW(**opt_kw)
else:
    rms_kw = {"learning_rate": args.lr, "rho": 0.9}
    if args.clipnorm > 0:
        rms_kw["global_clipnorm"] = args.clipnorm
    optimizer = tf.keras.optimizers.RMSprop(**rms_kw)
loss_fn = make_loss(args.lambda_cc, args.lambda_psd, args.lambda_band, fs)

run_config = {"model": net_name, "optimizer": args.optimizer, "lr": args.lr, "weight_decay": args.weight_decay,
              "clipnorm": args.clipnorm, "lr_schedule": args.lr_schedule, "lambda_cc": args.lambda_cc,
              "lambda_psd": args.lambda_psd, "lambda_band": args.lambda_band, "arch": args.arch, "arch_cfg": arch_cfg, "combin_num": args.combin_num, "independent_eval": not args.legacy_eval, "fresh_noise": args.fresh_noise,
              "base": args.base or (46 if args.arch == "pf" else 36), "blocks": args.blocks, "revin": args.revin,
              "temporal_att": not args.no_temporal_att, "freq_att": not args.no_freq_att,
              "snr_convention": args.snr_convention, "snr_range": [args.snr_low, args.snr_high],
              "noise_type": args.noise_type, "bandpass": args.bandpass, "quick": args.quick,
              "params": int(model.count_params())}

# ---------------------------------------------------------------- train
epoch_fn = None
if args.fresh_noise:
    if args.legacy_eval:
        raise SystemExit("--fresh-noise needs the independent validation/test stream (do not combine with --legacy-eval)")
    _fn = info["train_epoch_fn"]
    epoch_fn = (lambda ep: tuple(a[:len(noiseEEG_train)] for a in _fn(ep))) if args.quick else _fn
saved_model, history = train(
    model, noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, epochs, args.batch_size, optimizer,
    net_name, result_location, foldername, train_num,
    seed=args.seed, patience=args.patience, training_mode=True, run_config=run_config,
    fs=fs, lr_schedule=args.lr_schedule, base_lr=args.lr, loss_fn=loss_fn,
    n_levels=1 if args.quick else info["n_levels"], epoch_data_fn=epoch_fn)

# ---------------------------------------------------------------- save test signals + metrics
out_dir = os.path.join(run_dir, "nn_output")
os.makedirs(out_dir, exist_ok=True)
den = predict(saved_model, noiseEEG_test, net_name)
np.save(os.path.join(out_dir, "Denoiseoutput_test.npy"), den)
np.save(os.path.join(out_dir, "noiseinput_test.npy"), noiseEEG_test)
np.save(os.path.join(out_dir, "EEG_test.npy"), EEG_test)
np.save(os.path.join(out_dir, "loss_history.npy"), history)

nl, levels = info["n_levels"], info["snr_levels"]
summary = summarize(den, EEG_test, noiseEEG_test, fs=fs, n_levels=nl, snr_levels=levels)
summary["run"] = {**run_config, "seed": args.seed, "epochs_run": len(history["loss"]["val_mse"]),
                  "best_epoch": history["best_epoch"], "stopped_early": history["stopped_early"],
                  "batch_size": args.batch_size, "network": net_name}

# per-artifact results (blocks keep equal size because the noise type depends on the segment, not the block)
lab = np.asarray(info["test_noise_label"])
if args.noise_type == "MIXED" and not args.quick:
    for k, name in ((0, "EOG"), (1, "EMG")):
        sel = lab == k
        summary["by_noise_" + name] = summarize(den[sel], EEG_test[sel], noiseEEG_test[sel], fs=fs,
                                                n_levels=nl, snr_levels=levels)
with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print("%s | best epoch %d of %d run | %s parameters" % (
    foldername, history["best_epoch"], summary["run"]["epochs_run"], "{:,}".format(run_config["params"])))
print(format_summary(summary, args.snr_convention))
for name in ("EOG", "EMG"):
    s = summary.get("by_noise_" + name)
    if s:
        o = s["overall"]
        print("  %s only: CC %.4f  RRMSE-t %.4f  PSNR %.2f  SNR %.2f  RRMSE total %.4f" % (
            name, o["cc"], o["rrmse_t"], o["psnr"], o["snr"], o["rrmse_total"]))
print("=" * 70)
print("outputs in", out_dir)
