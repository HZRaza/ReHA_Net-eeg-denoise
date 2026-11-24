Fixed EEGdenoiseNet baseline pipeline
=====================================

Copy these into  EEGdenoiseNet-master/code/benchmark_networks/  and overwrite the old
data_prepare.py, train_method.py and main.py.  New: eeg_utils.py, tests/.
Network_structure.py, loss_function.py and save_method.py stay as they are.

1) Edit DATA_DIR and RESULT_DIR at the top of main.py (or pass --data-dir / --result-dir).
2) Smoke test first (a few minutes, own folder):      python main.py --quick
3) Full baseline:                                      python main.py
   - Ctrl+C is safe; run the same command again to resume.
   - Use a NEW --foldername for every new experiment (the old EMG_unet112dense_10_rmsp_test
     folder holds the run that mixed splits; do not reuse it).

Useful options
  --snr-convention repo|standard   repo = Zhang eq.2 / original code (nominal dB is really 2x by the
                                   usual definition); standard = usual 20*log10 definition
  --original-train-mode            model(x, training=False) while training, like the old code
  --patience 0                     no early stopping (always run all --epochs)
  --optimizer rmsp|adam|sgd        rmsp matches the repo default; the paper text says Adam
  --seed N                         different split / init

Outputs  results/<foldername>/1/
  nn_output/  noiseinput_test.npy, Denoiseoutput_test.npy, EEG_test.npy, loss_history.npy,
              metrics_summary.json (overall + per-SNR)
  checkpoint/state.pkl  (has val_cc_history / val_rrmse_history for plot_metric_epochs.py)
  best.weights.h5, split_info.npz (seed + segment indices of the split)

Re-compute every metric from saved outputs:   python eeg_utils.py results/<foldername>/1/nn_output
Self-checks (no TensorFlow needed):            python tests/test_data_and_metrics.py
