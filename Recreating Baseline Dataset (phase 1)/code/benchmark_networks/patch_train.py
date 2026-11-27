"""Applies the logging extension to Claude 2's train_method.py / main.py (fails loudly if a pattern is missing)."""
import sys
src_dir, dst_dir = sys.argv[1], sys.argv[2]
def sub(s, old, new, count=1):
    assert s.count(old) == count, f"pattern not found {s.count(old)}x (expected {count}):\n{old[:90]}"
    return s.replace(old, new)

t = open(f"{src_dir}/train_method.py").read()
t = sub(t, "import time\n\nimport numpy", "import time\nimport csv\nimport json\n\nimport numpy")
t = sub(t, "from eeg_utils import EarlyStopper, cc, config_fingerprint, epoch_permutation, rrmse_temporal\n",
        "from eeg_utils import EarlyStopper, cc, config_fingerprint, epoch_permutation, rrmse_temporal\nfrom metrics_full import full_metrics\n")
t = sub(t, "    return loss, tf.norm(grads[0])\n", "    return loss, tf.norm(grads[0]), tf.linalg.global_norm(grads)\n")
t = sub(t, "seed=0, patience=10, min_delta=0.0, training_mode=True, run_config=None):",
        "seed=0, patience=10, min_delta=0.0, training_mode=True, run_config=None,\n"
        "          noiseEEG_test=None, EEG_test=None, fs=256, n_levels=10, save_val_preds_at=(1, 2, 5, 10, 20, 30, 40, 50)):")
t = sub(t, "    y_val = np.asarray(EEG_val, dtype=np.float32).reshape(len(EEG_val), -1)\n",
        "    y_val = np.asarray(EEG_val, dtype=np.float32).reshape(len(EEG_val), -1)\n"
        "    y_test = None if EEG_test is None else np.asarray(EEG_test, dtype=np.float32).reshape(len(EEG_test), -1)\n")
t = sub(t, '    hist = {"train_mse": [], "train_mse_eval": [], "val_mse": [], "grads": [], "val_cc": [], "val_rrmse": []}\n',
'''    # per-epoch history. The first six keys are the originals (older plot scripts read them); the rest are new.
    SCALARS = ["train_mse", "train_mse_eval", "val_mse", "grads", "val_cc", "val_rrmse",
               "train_cc_eval", "train_rrmse_eval", "grad_norm_all", "lr", "epoch_time",
               "val_rrmse_f", "val_psnr", "val_snr", "val_reha_rrmse", "val_reha_rrmse_t", "val_reha_rrmse_s",
               "val_reha_rrmse_f", "val_cc_sd", "val_rrmse_sd",
               "test_mse", "test_cc", "test_rrmse", "test_rrmse_f", "test_psnr", "test_snr", "test_reha_rrmse"]
    LISTS = ["val_cc_by_snr", "val_rrmse_by_snr", "val_psnr_by_snr", "val_snr_by_snr"]
    hist = {k: [] for k in SCALARS + LISTS}
    os.makedirs(os.path.join(run_dir, "val_preds"), exist_ok=True)
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        with open(cfg_path, "w") as f:
            json.dump(dict(run_config or {}, seed=seed, batch_size=batch_size, network=denoise_network,
                           training_mode=bool(training_mode), epochs=epochs, patience=patience, min_delta=min_delta,
                           n_train=n, n_val=int(len(EEG_val)), n_test=0 if y_test is None else int(len(y_test))),
                      f, indent=2, default=str)
''')
t = sub(t, '        for k in hist:\n            hist[k] = list(s["hist"][k])\n',
        '        for k in hist:   # runs started with the older code lack the new keys: pad with NaN\n'
        '            hist[k] = list(s["hist"].get(k, [float("nan")] * start_epoch))\n')
t = sub(t, "                run_loss, run_grad = 0.0, 0.0\n", "                run_loss, run_grad, run_gall = 0.0, 0.0, 0.0\n")
t = sub(t, "                        loss, gnorm = train_step(", "                        loss, gnorm, gall = train_step(")
t = sub(t, "                        run_grad += float(gnorm)\n", "                        run_grad += float(gnorm)\n                        run_gall += float(gall)\n")
t = sub(t, "                grad_norm = run_grad / batch_num\n", "                grad_norm = run_grad / batch_num\n                grad_all = run_gall / batch_num\n")

a = t.index("                # training loss in inference mode")
b = t.index("                improved, stop = stopper.update")
new_block = '''                # training loss in inference mode (comparable with validation loss), fixed subset
                tr_out = predict(model, noiseEEG[:m_sub], denoise_network)
                train_mse_eval = float(np.mean((tr_out - y_all[:m_sub]) ** 2))
                train_cc_eval = float(np.mean(cc(tr_out, y_all[:m_sub])))
                train_rr_eval = float(np.mean(rrmse_temporal(tr_out, y_all[:m_sub])))

                # validation: the full metric bundle (this drives best-epoch / early stopping through val_mse)
                val_out = predict(model, noiseEEG_val, denoise_network)
                vm = full_metrics(val_out, y_val, fs, n_levels)
                val_mse, val_cc, val_rr = vm["mse"], vm["cc"], vm["rrmse_t"]

                tm = {}   # test metrics are logged only; they never influence training or the best epoch
                if y_test is not None:
                    tm = full_metrics(predict(model, noiseEEG_test, denoise_network), y_test, fs, n_levels)

                if (epoch + 1) in tuple(save_val_preds_at):
                    np.save(os.path.join(run_dir, "val_preds", "val_pred_epoch%d.npy" % (epoch + 1)), val_out)
                try:
                    cur_lr = float(np.asarray(optimizer.learning_rate))
                except Exception:
                    cur_lr = float("nan")
                nan = float("nan")
                rec = {"train_mse": train_mse, "train_mse_eval": train_mse_eval, "val_mse": val_mse, "grads": grad_norm,
                       "val_cc": val_cc, "val_rrmse": val_rr, "train_cc_eval": train_cc_eval,
                       "train_rrmse_eval": train_rr_eval, "grad_norm_all": grad_all, "lr": cur_lr,
                       "epoch_time": time.time() - t0, "val_rrmse_f": vm["rrmse_f"], "val_psnr": vm["psnr"],
                       "val_snr": vm["snr"], "val_reha_rrmse": vm["reha_rrmse"], "val_reha_rrmse_t": vm["reha_rrmse_t"],
                       "val_reha_rrmse_s": vm["reha_rrmse_s"], "val_reha_rrmse_f": vm["reha_rrmse_f"],
                       "val_cc_sd": vm["cc_sd"], "val_rrmse_sd": vm["rrmse_t_sd"],
                       "test_mse": tm.get("mse", nan), "test_cc": tm.get("cc", nan), "test_rrmse": tm.get("rrmse_t", nan),
                       "test_rrmse_f": tm.get("rrmse_f", nan), "test_psnr": tm.get("psnr", nan),
                       "test_snr": tm.get("snr", nan), "test_reha_rrmse": tm.get("reha_rrmse", nan),
                       "val_cc_by_snr": vm["cc_by_snr"], "val_rrmse_by_snr": vm["rrmse_t_by_snr"],
                       "val_psnr_by_snr": vm["psnr_by_snr"], "val_snr_by_snr": vm["snr_by_snr"]}
                for k, v in rec.items():
                    hist[k].append(v if k in LISTS else float(v))
                with open(os.path.join(run_dir, "epoch_log.csv"), "w", newline="") as f:   # readable copy of the scalars
                    w = csv.writer(f); w.writerow(["epoch"] + SCALARS)
                    for i in range(len(hist["val_mse"])):
                        w.writerow([i + 1] + ["%.6g" % hist[k][i] for k in SCALARS])

                with train_log.as_default():
                    tf.summary.scalar("loss", train_mse, step=epoch)
                with val_log.as_default():
                    tf.summary.scalar("loss", val_mse, step=epoch)
                    tf.summary.scalar("cc", val_cc, step=epoch)

'''
t = t[:a] + new_block + t[b:]
t = sub(t, '''                print("Epoch #: {}/{}, time {:.1f}s | train_mse {:.5f} (eval-mode {:.5f}) | "
                      "val_mse {:.5f}  val_CC {:.4f}  val_RRMSE {:.4f} | grad {:.4f}{}".format(
                          epoch + 1, epochs, time.time() - t0, train_mse, train_mse_eval, val_mse, val_cc,
                          val_rr, grad_norm, "  <- best" if improved else ""))''',
'''                print("Epoch #: {}/{}, time {:.1f}s | train_mse {:.5f} (eval-mode {:.5f}) | "
                      "val_mse {:.5f}  val_CC {:.4f}  val_RRMSE {:.4f}  PSNR {:.2f}  SNR {:.2f} | grad {:.4f}{}".format(
                          epoch + 1, epochs, time.time() - t0, train_mse, train_mse_eval, val_mse, val_cc,
                          val_rr, vm["psnr"], vm["snr"], grad_norm, "  <- best" if improved else ""))''')

t = sub(t, '    history = {\n        "grads": {"mse": hist["grads"]},',
'''    # model size, speed and training cost (for the "accuracy vs size / time" plots)
    try:
        xs = noiseEEG_val[:1024]
        predict(model, xs, denoise_network)                       # warm-up
        t1 = time.time(); predict(model, xs, denoise_network); ms = (time.time() - t1) / len(xs) * 1000
        with open(os.path.join(run_dir, "model_info.json"), "w") as f:
            json.dump({"params_total": int(model.count_params()),
                       "params_trainable": int(sum(int(np.prod(v.shape)) for v in model.trainable_variables)),
                       "inference_ms_per_segment_this_machine": ms,
                       "train_time_s": float(np.nansum(hist["epoch_time"])), "epochs_run": len(hist["val_mse"]),
                       "best_epoch": stopper.best_epoch, "stopped_early": bool(stopped_early)}, f, indent=2)
    except Exception as e:
        print("could not write model_info.json:", e)

    history = {
        "grads": {"mse": hist["grads"]},''')
t = sub(t, '        "metrics": {"val_cc": hist["val_cc"], "val_rrmse": hist["val_rrmse"]},',
        '        "metrics": {k: hist[k] for k in hist if k not in ("train_mse", "train_mse_eval", "val_mse", "grads")},')
open(f"{dst_dir}/train_method.py", "w").write(t)

m = open(f"{src_dir}/main.py").read()
m = sub(m, "seed=args.seed, patience=args.patience, training_mode=not args.original_train_mode, run_config=run_config)",
        "seed=args.seed, patience=args.patience, training_mode=not args.original_train_mode, run_config=run_config,\n"
        "    noiseEEG_test=noiseEEG_test, EEG_test=EEG_test, fs=fs, n_levels=info[\"n_levels\"])")
open(f"{dst_dir}/main.py", "w").write(m)
print("patched OK")
