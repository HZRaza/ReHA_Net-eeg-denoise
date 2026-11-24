PHASE 2: ReHA-Net recreation (Keras)
====================================
Copy these files into your benchmark_networks folder (same place as main.py). train_method.py REPLACES the
old one (only change: optional loss_fn argument; Phase 1 behaviour is identical when it is not given).

  data_reha.py      dataset: EEG + EOG/EMG, SNR -5..20 dB, 10x mixing, split before mixing
  model_reha.py     ReHA-Net (U-Net + MSC blocks + hybrid attention + ReVIN) and the composite loss
  main_reha.py      runner (edit DATA_DIR / RESULT_DIR at the top, like main.py)
  train_method.py   updated trainer
  tests/test_reha_data.py

Run, in this order, from the benchmark_networks folder with the .venv active:
  python tests/test_reha_data.py                 # numpy-only checks of the data generator
  python main_reha.py --summary                  # builds the model, prints layers + parameter count
  python main_reha.py --quick                    # 2-epoch smoke test (small subset)
  python main_reha.py                            # full run: MIXED (EOG+EMG), 40 epochs, patience 8
  python main_reha.py --noise-type EOG --foldername reha_EOG     # EOG only, to compare with Table 5
Plots:  python plot_results.py <results>\reha_MIXED_baseline\1 --snr-convention standard

Defaults follow the paper: AdamW 5e-5, weight decay 1e-4, cosine annealing with warm restarts (T0=10, Tmult=2),
batch 32, 40 epochs, patience 8, grad-norm clip 1.0, loss = MSE + 0.05*(1-CC) + 0.05*MAE(|FFT|).

Things I had to decide because the paper does not say (all are command-line switches):
  * SNR convention: 'standard' (real dB). Zhang's repo uses 'repo' (real dB = 2x nominal). Pick one and keep it.
  * Test blocks: -5, 0, 5, 10, 15, 20 dB (--n-levels 6). MIXED = 50% EOG / 50% EMG per segment.
  * Hybrid attention internals, ReVIN without reversal (--revin reverse adds it), width (--base 36 ~ 2.1 M params).
  * lambda values: paper states both 0.05/0.05 and 0.5/0.05 (--lambda-cc, --lambda-psd).
Logged "train_mse" in this run is the composite training loss; "train_mse_eval" and val_mse stay plain MSE,
so validation/early stopping are comparable with Phase 1.
Ablations: --no-temporal-att, --no-freq-att, --revin none, --lambda-cc 0 --lambda-psd 0, --blocks 1.

NOT run on real TensorFlow in my sandbox (none installed). The data generator and the Phase 1 trainer logic are
tested; the Keras model is not. If --summary errors, send me the traceback.
