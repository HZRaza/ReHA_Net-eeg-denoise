"""ReHA-Net style dataset (Phase 2): EEG + EOG and/or EMG, SNR -5..20 dB, 10x combinatorial mixing.

Built on the same EEGdenoiseNet clean-EEG / EOG / EMG segments (all resampled to 256 Hz, 512 samples).
Differences from data_prepare.prepare_data (the Phase 1 / Zhang dataset):
  * wider SNR range (default -5..20 dB) and 6 test blocks (-5, 0, 5, 10, 15, 20)
  * noise_type 'MIXED': every pair gets either an EOG or an EMG segment (50/50), so one model is
    trained on both artifacts
  * the split is done on the CLEAN EEG and on each noise pool BEFORE any mixing, so no EEG or noise
    segment of the test set ever appears in training
  * default SNR convention is 'standard' (20*log10 of the RMS ratio = real dB). The paper does not
    say which convention it uses; see README_phase2.txt.
  * optional 0.5-40 Hz band-pass (applied to EEG and noise before mixing)

Returned tuple has the same layout as prepare_data, so main-style code can be reused.
"""
import numpy as np

from data_prepare import SNR_CONVENTIONS, _standardise, mix

NOISE_TYPES = ("EOG", "EMG", "MIXED")


def bandpass(x, fs=256, lo=0.5, hi=40.0, order=4):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def load_segments(data_dir, noise_type="MIXED"):
    """Load EEG, EOG, EMG as (N, 512) float arrays at 256 Hz. EMG falls back to the 512 Hz file."""
    import os
    eeg = np.load(os.path.join(data_dir, "EEG_all_epochs.npy"))
    eog = emg = None
    if noise_type in ("EOG", "MIXED"):
        eog = np.load(os.path.join(data_dir, "EOG_all_epochs.npy"))
    if noise_type in ("EMG", "MIXED"):
        p256 = os.path.join(data_dir, "EMG_all_epochs.npy")
        if os.path.exists(p256):
            emg = np.load(p256)
        else:  # 1024-sample file at 512 Hz -> 512 samples at 256 Hz
            from scipy.signal import resample_poly
            emg = resample_poly(np.load(os.path.join(data_dir, "EMG_all_epochs_512hz.npy")), 1, 2, axis=-1)
    for name, a in (("EEG", eeg), ("EOG", eog), ("EMG", emg)):
        if a is not None and a.shape[1] != 512:
            raise ValueError("%s segments have %d samples, expected 512 (256 Hz, 2 s)" % (name, a.shape[1]))
    return eeg, eog, emg


def _split_sizes(n, train_per):
    tr = round(train_per * n)
    va = round((n - tr) / 2)
    return tr, va, n - tr - va


def prepare_reha_data(EEG_all, EOG_all=None, EMG_all=None, noise_type="MIXED", combin_num=10, train_per=0.8,
                      seed=0, snr_convention="standard", snr_range=(-5.0, 20.0), n_levels=6,
                      use_bandpass=False, fs=256, verbose=True, independent_eval=False):
    """Returns (noisy_train, clean_train, noisy_val, clean_val, noisy_test, clean_test, test_std, info).

    info["test_noise_label"] / info["val_noise_label"]: 0 = EOG, 1 = EMG for every row (blocks repeated).
    independent_eval=False (legacy, used by the first run): validation/test noise is drawn from the same random
    stream as the training data, so it changes with combin_num.  independent_eval=True: validation and test use
    their own stream (seed + 12345), so they are identical for any combin_num -- required to compare protocols.
    """
    if snr_convention not in SNR_CONVENTIONS:
        raise ValueError("snr_convention must be one of %s" % (SNR_CONVENTIONS,))
    if noise_type not in NOISE_TYPES:
        raise ValueError("noise_type must be one of %s" % (NOISE_TYPES,))
    pools = {}
    if noise_type in ("EOG", "MIXED"):
        pools[0] = np.asarray(EOG_all, dtype=np.float64)
    if noise_type in ("EMG", "MIXED"):
        pools[1] = np.asarray(EMG_all, dtype=np.float64)
    eeg_all = np.asarray(EEG_all, dtype=np.float64)
    if use_bandpass:
        eeg_all = bandpass(eeg_all, fs)
        pools = {k: bandpass(v, fs) for k, v in pools.items()}

    rng = np.random.RandomState(seed)
    lo, hi = float(snr_range[0]), float(snr_range[1])
    levels = np.linspace(lo, hi, n_levels)
    kinds = sorted(pools)

    # ---- split every pool on its own, before mixing ---------------------------------------
    def split(a):
        idx = rng.permutation(len(a))
        tr, va, te = _split_sizes(len(a), train_per)
        return idx[:tr], idx[tr:tr + va], idx[tr + va:]

    eeg_tr, eeg_va, eeg_te = split(eeg_all)
    noise_split = {k: split(pools[k]) for k in kinds}   # k -> (train_idx, val_idx, test_idx)

    rng_eval = np.random.RandomState(seed + 12345) if independent_eval else rng

    def draw_noise(part, n, rr=None):
        """n noise segments for split `part` (0 train,1 val,2 test). Returns (segments, labels). rr = optional own RNG."""
        r = rr if rr is not None else (rng if part == 0 else rng_eval)
        labels = np.array([kinds[i % len(kinds)] for i in range(n)]) if part else \
            np.array([kinds[j] for j in r.randint(0, len(kinds), n)])
        out = np.empty((n, eeg_all.shape[1]))
        for k in kinds:
            m = labels == k
            idx = noise_split[k][part]
            out[m] = pools[k][idx[r.randint(0, len(idx), int(m.sum()))]]
        return out, labels

    # ---- training: combin_num independent re-pairings, SNR uniform in range -----------------
    eeg_train = np.concatenate([eeg_all[eeg_tr[rng.permutation(len(eeg_tr))]] for _ in range(combin_num)])
    noise_train, train_label = draw_noise(0, len(eeg_train))
    snr_train = rng.uniform(lo, hi, len(eeg_train))
    noisy_train, eeg_train, _ = _standardise(mix(eeg_train, noise_train, snr_train, snr_convention), eeg_train)

    def train_epoch(epoch):
        """FRESH training data for one epoch: new EEG<->noise pairings, new noise segments and new SNRs, drawn from the
        training split only (same size as the fixed set: combin_num x n_train). Deterministic in (seed, epoch), so a
        resumed run sees exactly the data it would have seen. Validation/test are untouched (independent_eval stream)."""
        r = np.random.RandomState((int(seed) * 1000003 + 7919 * (int(epoch) + 1)) % (2 ** 31 - 1))
        e = np.concatenate([eeg_all[eeg_tr[r.permutation(len(eeg_tr))]] for _ in range(combin_num)])
        nz, _ = draw_noise(0, len(e), r)
        sn = r.uniform(lo, hi, len(e))
        noisy, clean, _ = _standardise(mix(e, nz, sn, snr_convention), e)
        return noisy.astype(np.float32), clean.astype(np.float32)

    # ---- val / test: fixed pairing, one block per SNR level, same segments in every block ---
    def blocks(part, eeg_idx):
        eeg = eeg_all[eeg_idx]
        noise, lab = draw_noise(part, len(eeg))
        noisy = np.concatenate([mix(eeg, noise, lv, snr_convention) for lv in levels])
        clean = np.tile(eeg, (n_levels, 1))
        noisy, clean, std = _standardise(noisy, clean)
        return noisy, clean, std, np.tile(lab, n_levels)

    noisy_val, eeg_val, _, val_label = blocks(1, eeg_va)
    noisy_test, eeg_test, test_std, test_label = blocks(2, eeg_te)

    info = {
        "seed": int(seed), "snr_convention": snr_convention, "snr_range": (lo, hi), "snr_levels": levels,
        "n_levels": int(n_levels), "noise_type": noise_type, "bandpass": bool(use_bandpass),
        "n_train_segments": int(len(eeg_tr)), "n_val_segments": int(len(eeg_va)), "n_test_segments": int(len(eeg_te)),
        "eeg_idx": np.concatenate([eeg_tr, eeg_va, eeg_te]),
        "noise_idx": np.array([-1]), "test_noise_label": test_label, "val_noise_label": val_label,
        "train_noise_label": train_label, "train_epoch_fn": train_epoch,
    }
    if verbose:
        real = (2 * lo, 2 * hi) if snr_convention == "repo" else (lo, hi)
        print("EEG segments train/val/test: %d / %d / %d  (seed %d)" % (len(eeg_tr), len(eeg_va), len(eeg_te), seed))
        for k in kinds:
            print("  %s pool: %d / %d / %d" % (("EOG", "EMG")[k], *(len(i) for i in noise_split[k])))
        print("noise_type %s | SNR convention '%s': nominal %.1f..%.1f dB -> real %.1f..%.1f dB, test levels %s"
              % (noise_type, snr_convention, lo, hi, real[0], real[1], np.round(levels, 1).tolist()))
        print("train", noisy_train.shape, "| val", noisy_val.shape, "| test", noisy_test.shape)
    f32 = lambda a: a.astype(np.float32)
    return (f32(noisy_train), f32(eeg_train), f32(noisy_val), f32(eeg_val),
            f32(noisy_test), f32(eeg_test), test_std, info)


def save_split_info(info, path):
    np.savez(path, seed=info["seed"], snr_convention=info["snr_convention"], snr_range=np.array(info["snr_range"]),
             snr_levels=info["snr_levels"], eeg_idx=info["eeg_idx"], noise_idx=info["noise_idx"],
             n_train=info["n_train_segments"], n_val=info["n_val_segments"], n_test=info["n_test_segments"],
             test_noise_label=info["test_noise_label"], noise_type=info["noise_type"])
