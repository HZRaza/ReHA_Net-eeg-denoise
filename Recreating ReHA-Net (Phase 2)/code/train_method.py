import math
import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from eeg_utils import EarlyStopper, config_fingerprint, epoch_permutation, get_lr, metric_means

# Author of the original: Haoming Zhang
# Rewritten training process.  What changed compared with the original train_method.py:
#   * real mini-batches (one forward/backward pass per batch instead of one per sample)
#   * model(x, training=True): BatchNorm statistics update and dropout is active while training
#     (set training_mode=False to get the original behaviour back)
#   * the data split is fixed by a seed in data_prepare; resume refuses to continue if the
#     settings changed, so a resumed run can no longer mix up train/validation/test
#   * every epoch logs validation CC, RRMSE (Zhang), PSNR, SNR and the ReHA-Net RRMSE variants
#     (t / f / s / total); state.pkl keeps val_cc_history / val_rrmse_history for plot_metric_epochs.py
#   * optional learning-rate schedule (constant, step, exp, cosine, cosine_restarts); lr is logged
#   * best-validation-epoch weights are saved and restored at the end (original returned the
#     last-epoch weights), optional early stopping
#   * checkpoint + resume after Ctrl+C or a crash, exact because shuffling depends on (seed, epoch)


@tf.function
def train_step(model, x, y, optimizer, training, loss_fn=None):
    """One optimisation step on a whole batch. x: (B,T[,1]), y: (B,T).
    Plain MSE unless loss_fn(y, pred) is given (used by ReHA-Net's composite loss)."""
    with tf.GradientTape() as tape:
        pred = model(x, training=training)
        pred = tf.reshape(pred, tf.shape(y))
        loss = tf.reduce_mean(tf.square(pred - y)) if loss_fn is None else loss_fn(y, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, tf.norm(grads[0])


def _prep_input(x, denoise_network):
    """Add the channel axis for the conv/rnn models. denoise_network=None leaves x untouched
    (the original code fed 2-D arrays at validation/test time)."""
    x = np.asarray(x, dtype=np.float32)
    if denoise_network is None or denoise_network == "fcNN":
        return x
    return x[..., None]


def predict(model, x, denoise_network=None, chunk=512):
    """Inference-mode output for many segments, in chunks to bound memory. Returns (N, T) float32."""
    x = _prep_input(x, denoise_network)
    outs = []
    for i in range(0, len(x), chunk):
        part = x[i:i + chunk]
        out = model(part, training=False)
        outs.append(np.asarray(out, dtype=np.float32).reshape(len(part), -1))
    return np.concatenate(outs, axis=0)


def test_step(model, noiseEEG_test, EEG_test, denoise_network=None):
    """Same name/signature as the original (used by save_method.save_eeg). Returns (output, mse)."""
    out = predict(model, noiseEEG_test, denoise_network)
    y = np.asarray(EEG_test, dtype=np.float32).reshape(out.shape)
    return out, float(np.mean((out - y) ** 2))


def _atomic_pickle(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def train(model, noiseEEG, EEG, noiseEEG_val, EEG_val, epochs, batch_size, optimizer,
          denoise_network, result_location, foldername, train_num,
          seed=0, patience=10, min_delta=0.0, training_mode=True, run_config=None,
          fs=256, lr_schedule="constant", base_lr=None, loss_fn=None, n_levels=1, epoch_data_fn=None):
    """Train, checkpoint every epoch, keep the best-validation weights.

    The first 11 arguments match the original train(); the rest are keyword options.
    patience <= 0 disables early stopping (the best epoch is still tracked and restored).
    Returns (model_with_best_weights, history).
    """
    run_dir = os.path.join(result_location, foldername, train_num)
    ckpt_dir = os.path.join(run_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best.weights.h5")
    state_file = os.path.join(ckpt_dir, "state.pkl")

    n = int(noiseEEG.shape[0])
    batch_num = math.ceil(n / batch_size)
    x_all = _prep_input(noiseEEG, denoise_network)
    y_all = np.asarray(EEG, dtype=np.float32).reshape(n, -1)
    y_val = np.asarray(EEG_val, dtype=np.float32).reshape(len(EEG_val), -1)
    m_sub = min(n, 3400)  # training subset scored in inference mode, comparable with validation

    # settings that must not change between a run and its resume (epochs/patience may)
    fingerprint = config_fingerprint(dict(run_config or {}, seed=seed, batch_size=batch_size,
                                          network=denoise_network, training_mode=bool(training_mode),
                                          n_train=n, n_val=int(len(EEG_val)),
                                          lr_schedule=lr_schedule, base_lr=base_lr))

    train_log = tf.summary.create_file_writer(os.path.join(run_dir, "train"))
    val_log = tf.summary.create_file_writer(os.path.join(run_dir, "test"))

    # ---------------- checkpoint / resume ----------------
    try:  # create optimizer slots up front so the checkpoint restore is exact
        optimizer.build(model.trainable_variables)
    except Exception:
        pass
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=2)

    hist_keys = ["train_mse", "train_mse_eval", "val_mse", "grads", "val_cc", "val_rrmse", "val_psnr", "val_snr",
                 "val_rrmse_t", "val_rrmse_f", "val_rrmse_s", "val_rrmse_total", "val_rrmse_spectral", "lr", "gain_levels"]
    hist = {k: [] for k in hist_keys}
    stopper = EarlyStopper(patience, min_delta)
    start_epoch, finished = 0, False

    if os.path.exists(state_file):
        with open(state_file, "rb") as f:
            s = pickle.load(f)
        if s.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "The checkpoint in '%s' was made with different settings (seed, batch size, network, "
                "data size, SNR convention, optimizer or training mode). Use a new --foldername, or delete "
                "that folder, instead of resuming." % run_dir)
        start_epoch = s["next_epoch"]
        finished = s.get("finished", False)
        for k in hist:
            hist[k] = list(s["hist"].get(k, []))
        stopper = EarlyStopper.from_state(s["stopper"], patience, min_delta)
        if start_epoch > 0 and not finished:
            prefix = os.path.join(ckpt_dir, "ckpt-%d" % start_epoch)
            if not os.path.exists(prefix + ".index"):
                raise RuntimeError("state.pkl says epoch %d is done but checkpoint '%s' is missing."
                                   % (start_epoch, prefix))
            ckpt.restore(prefix)
        print("Resuming from epoch %d/%d%s" % (start_epoch + 1, epochs, " (already finished)" if finished else ""))

    def write_state(next_epoch, is_finished):
        _atomic_pickle({
            "fingerprint": fingerprint, "next_epoch": next_epoch, "finished": is_finished,
            "hist": hist, "stopper": stopper.state(),
            # keys kept for the older plotting scripts
            "train_mse_history": hist["train_mse"], "val_mse_history": hist["val_mse"],
            "mse_grads_history": hist["grads"], "val_cc_history": hist["val_cc"],
            "val_rrmse_history": hist["val_rrmse"], "val_mse_min": stopper.best,
            "best_epoch": stopper.best_epoch,
        }, state_file)

    # ---------------- training loop ----------------
    epoch = start_epoch
    stopped_early = finished
    try:
        if not finished:
            for epoch in range(start_epoch, epochs):
                t0 = time.time()
                lr_now = None
                if base_lr is not None and lr_schedule != "constant":
                    lr_now = get_lr(lr_schedule, epoch, epochs, base_lr)
                    optimizer.learning_rate.assign(lr_now)
                else:
                    lr_now = float(base_lr) if base_lr is not None else float("nan")
                if epoch_data_fn is not None:   # fresh noise pairing every epoch (same size as the fixed set)
                    noiseEEG, EEG = epoch_data_fn(epoch)
                    x_all = _prep_input(noiseEEG, denoise_network)
                    y_all = np.asarray(EEG, dtype=np.float32).reshape(len(EEG), -1)
                    assert len(y_all) == n, "epoch_data_fn must return %d rows" % n
                order = epoch_permutation(seed, epoch, n)
                run_loss, run_grad = 0.0, 0.0

                with tqdm(total=batch_num, position=0, leave=True) as pbar:
                    for b in range(batch_num):
                        idx = order[b * batch_size:(b + 1) * batch_size]
                        loss, gnorm = train_step(model, tf.constant(x_all[idx]), tf.constant(y_all[idx]),
                                                 optimizer, bool(training_mode), loss_fn)
                        run_loss += float(loss) * len(idx)
                        run_grad += float(gnorm)
                        pbar.update()
                train_mse = run_loss / n
                grad_norm = run_grad / batch_num

                # training loss in inference mode (comparable with validation loss), fixed subset
                tr_out = predict(model, noiseEEG[:m_sub], denoise_network)
                train_mse_eval = float(np.mean((tr_out - y_all[:m_sub]) ** 2))

                # validation
                val_out = predict(model, noiseEEG_val, denoise_network)
                val_mse = float(np.mean((val_out - y_val) ** 2))
                vm = metric_means(val_out, y_val, fs)
                val_cc, val_rr = vm["cc"], vm["rrmse_temporal"]

                # SNR gain (dB, power ratio) per validation SNR block: >0 = the network helped at that input level
                gl = []
                if n_levels > 1 and len(y_val) % n_levels == 0:
                    xv = np.asarray(noiseEEG_val, dtype=np.float32).reshape(len(y_val), -1)
                    mb = len(y_val) // n_levels
                    for li in range(n_levels):
                        sl = slice(li * mb, (li + 1) * mb)
                        gl.append(float(10 * np.log10(np.mean((xv[sl] - y_val[sl]) ** 2) / np.mean((val_out[sl] - y_val[sl]) ** 2))))
                hist["gain_levels"].append(gl)
                for k, v in (("train_mse", train_mse), ("train_mse_eval", train_mse_eval), ("val_mse", val_mse),
                             ("grads", grad_norm), ("val_cc", val_cc), ("val_rrmse", val_rr),
                             ("val_psnr", vm["psnr"]), ("val_snr", vm["snr"]), ("val_rrmse_t", vm["rrmse_t"]),
                             ("val_rrmse_f", vm["rrmse_f"]), ("val_rrmse_s", vm["rrmse_s"]),
                             ("val_rrmse_total", vm["rrmse_total"]), ("val_rrmse_spectral", vm["rrmse_spectral"]),
                             ("lr", lr_now)):
                    hist[k].append(float(v))

                with train_log.as_default():
                    tf.summary.scalar("loss", train_mse, step=epoch)
                with val_log.as_default():
                    tf.summary.scalar("loss", val_mse, step=epoch)

                improved, stop = stopper.update(val_mse, epoch + 1)
                if improved:
                    model.save_weights(best_path)

                print("Epoch #: {}/{}, time {:.1f}s | train_mse {:.5f} (eval-mode {:.5f}) | "
                      "val_mse {:.5f}  CC {:.4f}  RRMSE {:.4f}  PSNR {:.2f}  SNR {:.2f} | grad {:.4f}{}".format(
                          epoch + 1, epochs, time.time() - t0, train_mse, train_mse_eval, val_mse, val_cc,
                          val_rr, vm["psnr"], vm["snr"], grad_norm, "  <- best" if improved else ""))
                if gl:
                    print("   gain by input level (dB): " + "  ".join("%+.1f" % g for g in gl))

                manager.save(checkpoint_number=epoch + 1)
                write_state(epoch + 1, stop)

                if stop:
                    stopped_early = True
                    print("Early stopping: no validation improvement for %d epochs. Best epoch: %d (val_mse %.5f)"
                          % (stopper.bad_epochs, stopper.best_epoch, stopper.best))
                    break
    except KeyboardInterrupt:
        print("\nInterrupted. The unfinished epoch is discarded. Run again with the same settings and "
              "foldername to resume from epoch %d." % (epoch + 1))
        sys.exit(0)

    # ---------------- restore the best-validation weights ----------------
    if os.path.exists(best_path):
        model.load_weights(best_path)
        print("Loaded best weights from epoch %d (val_mse %.5f)" % (stopper.best_epoch, stopper.best))
    else:
        print("WARNING: no best-epoch weights were saved (validation loss was never finite); "
              "using the final weights.")

    history = {
        "grads": {"mse": hist["grads"]},
        "loss": {"train_mse": hist["train_mse"], "train_mse_eval": hist["train_mse_eval"],
                 "val_mse": hist["val_mse"]},
        "metrics": {k: hist[k] for k in hist_keys if k.startswith("val_") and k != "val_mse"},
        "lr": hist["lr"], "gain_levels": hist["gain_levels"],
        "best_epoch": stopper.best_epoch,
        "stopped_early": bool(stopped_early),
    }
    return model, history
