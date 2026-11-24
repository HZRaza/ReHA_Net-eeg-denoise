import argparse
import json
import os

import numpy as np
import tensorflow as tf

from data_prepare import prepare_data, save_split_info
from eeg_utils import format_summary, summarize
from Network_structure import Complex_CNN, RNN_lstm, fcNN, simple_CNN
from save_method import save_eeg
from train_method import train

# EEGdenoiseNet baseline runner (fixed version).
# Original author: Haoming Zhang.  Edit the two paths below once, then e.g.
#
#   python main.py --quick                     # ~2 epoch smoke test on a small subset
#   python main.py                             # full baseline (EOG, Simple CNN, repo SNR convention)
#   python main.py --snr-convention standard   # same, but with the usual 20*log10 SNR definition
#
# Ctrl+C is safe: run the same command again to resume.

DATA_DIR = r"D:/People/hadiacode/New folder/EEGdenoiseNet-master/EEGdenoiseNet-master/data/"          # <- change if needed
RESULT_DIR = r"D:/People/hadiacode/New folder/EEGdenoiseNet-master/EEGdenoiseNet-master/results"     # <- change if needed

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true", help="smoke test: small subset, 2 epochs, own folder")
p.add_argument("--foldername", default="EOG_simpleCNN_baseline", help="new name for every new experiment")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--batch-size", type=int, default=40)
p.add_argument("--patience", type=int, default=10, help="early stopping patience in epochs, 0 = off")
p.add_argument("--network", default="Simple_CNN", choices=["fcNN", "Simple_CNN", "Complex_CNN", "RNN_lstm"])
p.add_argument("--noise-type", default="EOG", choices=["EOG", "EMG"])
p.add_argument("--optimizer", default="rmsp", choices=["rmsp", "adam", "sgd"])
p.add_argument("--snr-convention", default="repo", choices=["repo", "standard"])
p.add_argument("--snr-low", type=float, default=-7.0)
p.add_argument("--snr-high", type=float, default=2.0)
p.add_argument("--original-train-mode", action="store_true",
               help="call the model with training=False while training (what the original code did)")
p.add_argument("--data-dir", default=DATA_DIR)
p.add_argument("--result-dir", default=RESULT_DIR)
args = p.parse_args()

try:
    tf.keras.utils.set_random_seed(args.seed)
except AttributeError:  # very old TF
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

epochs, foldername = args.epochs, args.foldername
if args.quick:
    epochs, foldername = 2, foldername + "_quick"

datanum, fs = (512, 256) if args.noise_type == "EOG" else (1024, 512)
result_location = args.result_dir.replace("\\", "/").rstrip("/")
train_num = "1"

# ---------------------------------------------------------------- data
data_dir = args.data_dir
if args.noise_type == "EOG":
    EEG_all = np.load(os.path.join(data_dir, "EEG_all_epochs.npy"))
    noise_all = np.load(os.path.join(data_dir, "EOG_all_epochs.npy"))
else:
    EEG_all = np.load(os.path.join(data_dir, "EEG_all_epochs_512hz.npy"))
    noise_all = np.load(os.path.join(data_dir, "EMG_all_epochs_512hz.npy"))

(noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test,
 test_std_VALUE, info) = prepare_data(
    EEG_all=EEG_all, noise_all=noise_all, combin_num=10, train_per=0.8, noise_type=args.noise_type,
    seed=args.seed, snr_convention=args.snr_convention, snr_range=(args.snr_low, args.snr_high))

if args.quick:  # keep every SNR level in val/test, just fewer segments
    noiseEEG_train, EEG_train = noiseEEG_train[:1000], EEG_train[:1000]
    noiseEEG_val, EEG_val = noiseEEG_val[::10], EEG_val[::10]
    noiseEEG_test, EEG_test = noiseEEG_test[::10], EEG_test[::10]

run_dir = os.path.join(result_location, foldername, train_num)
os.makedirs(run_dir, exist_ok=True)
save_split_info(info, os.path.join(run_dir, "split_info.npz"))

# ---------------------------------------------------------------- model / optimizer
model = {"fcNN": fcNN, "Simple_CNN": simple_CNN, "Complex_CNN": Complex_CNN, "RNN_lstm": RNN_lstm}[args.network](datanum)

if args.optimizer == "rmsp":
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005, rho=0.9)
elif args.optimizer == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0002, momentum=0.9, nesterov=False)

# ---------------------------------------------------------------- train
run_config = {"optimizer": args.optimizer, "snr_convention": args.snr_convention,
              "snr_range": [args.snr_low, args.snr_high], "noise_type": args.noise_type, "quick": args.quick}

saved_model, history = train(
    model, noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, epochs, args.batch_size, optimizer,
    args.network, result_location, foldername, train_num,
    seed=args.seed, patience=args.patience, training_mode=not args.original_train_mode, run_config=run_config,
    noiseEEG_test=noiseEEG_test, EEG_test=EEG_test, fs=fs, n_levels=info["n_levels"])

# ---------------------------------------------------------------- save test signals + metrics
save_eeg(saved_model, result_location, foldername, False, False, True,
         noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, train_num=train_num)

out_dir = os.path.join(run_dir, "nn_output")
np.save(os.path.join(out_dir, "loss_history.npy"), history)

den = np.load(os.path.join(out_dir, "Denoiseoutput_test.npy"))
summary = summarize(den, EEG_test, noiseEEG_test, fs=fs, n_levels=info["n_levels"], snr_levels=info["snr_levels"])
summary["run"] = {**run_config, "seed": args.seed, "epochs_run": len(history["loss"]["val_mse"]),
                  "best_epoch": history["best_epoch"], "stopped_early": history["stopped_early"],
                  "batch_size": args.batch_size, "network": args.network,
                  "training_mode": not args.original_train_mode}
with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print("%s | best epoch %d of %d run" % (foldername, history["best_epoch"], summary["run"]["epochs_run"]))
print(format_summary(summary, args.snr_convention))
print("=" * 70)
print("outputs in", out_dir)
