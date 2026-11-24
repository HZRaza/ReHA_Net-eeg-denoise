import math

import numpy as np

# EEGdenoiseNet data preparation - seeded, vectorised rewrite of Haoming Zhang's data_prepare.py.
#
# What changed compared with the original:
#   * `seed`: the whole split / pairing / SNR draw is reproducible, so a restart or resume
#     sees exactly the same train / validation / test segments (no more train->test leakage).
#   * `snr_convention`: 'repo' reproduces the original behaviour, 'standard' uses the usual
#     SNR definition (see below).
#   * returns an extra `info` dict (seed, permutations, counts) so the split can be saved.
#   * vectorised numpy instead of per-segment python loops (much faster, identical maths).
#
# SNR conventions
#   'repo'     RMS(eeg) / RMS(lambda*noise) = 10**(snr_db/10)
#              i.e. SNR = 10*log10(RMS ratio).  This is Zhang et al. eq. (2) and what the
#              original code does.  By the usual definition the real SNR is 2x the nominal dB
#              (nominal -7..+2 dB is really about -14..+4 dB).
#   'standard' RMS(eeg) / RMS(lambda*noise) = 10**(snr_db/20)
#              i.e. SNR = 20*log10(RMS ratio) = 10*log10(power ratio).

SNR_CONVENTIONS = ("repo", "standard")


def get_rms(records):
    # kept for backwards compatibility with the original module
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def random_signal(signal, combin_num, rng=None):
    # Random disturb and augment signal (kept for compatibility; optional rng for reproducibility)
    if rng is None:
        rng = np.random
    random_result = []
    for _ in range(combin_num):
        random_num = rng.permutation(signal.shape[0])
        random_result.append(signal[random_num, :])
    return np.array(random_result)


def _rms(x):
    return np.sqrt(np.mean(np.square(x), axis=-1))


def snr_ratio(snr_db, convention):
    """RMS(eeg) / RMS(scaled noise) that realises `snr_db` under `convention`."""
    snr_db = np.asarray(snr_db, dtype=np.float64)
    if convention == "repo":
        return 10.0 ** (0.1 * snr_db)
    if convention == "standard":
        return 10.0 ** (0.05 * snr_db)
    raise ValueError("snr_convention must be one of %s, got %r" % (SNR_CONVENTIONS, convention))


def mix(eeg, noise, snr_db, convention):
    """noisy = eeg + lambda * noise.  eeg, noise: (N, T); snr_db: scalar or (N,)."""
    lam = _rms(eeg) / (_rms(noise) * snr_ratio(snr_db, convention))
    return eeg + noise * lam[:, None]


def _standardise(noisy, clean):
    """Divide both by the std of the noisy segment (Zhang et al. eq. 4)."""
    std = np.std(noisy, axis=-1, keepdims=True)
    return noisy / std, clean / std, std[:, 0]


def prepare_data(EEG_all, noise_all, combin_num, train_per, noise_type,
                 seed=0, snr_convention="repo", snr_range=(-7.0, 2.0), n_levels=10,
                 verbose=True):
    """Build standardised (noisy, clean) pairs for train / validation / test.

    Returns
      noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, test_std, info

    Train : combin_num independent re-pairings of the training segments, SNR uniform in snr_range.
    Val / test : n_levels equal blocks (lowest SNR first); every block holds the same segments.
    """
    if snr_convention not in SNR_CONVENTIONS:
        raise ValueError("snr_convention must be one of %s" % (SNR_CONVENTIONS,))
    rng = np.random.RandomState(seed)
    lo, hi = float(snr_range[0]), float(snr_range[1])

    n_eeg, n_noise = EEG_all.shape[0], noise_all.shape[0]
    eeg_idx = rng.permutation(n_eeg)
    noise_idx = rng.permutation(n_noise)

    if noise_type == "EMG":  # reuse some EEG segments to match the number of EMG segments
        reuse_num = n_noise - n_eeg
        eeg_idx = np.concatenate([eeg_idx[:reuse_num], eeg_idx])
    elif noise_type == "EOG":  # drop some EEG segments to match the number of EOG segments
        eeg_idx = eeg_idx[:n_noise]
    else:
        raise ValueError("noise_type must be 'EOG' or 'EMG'")

    eeg_r = EEG_all[eeg_idx]
    noise_r = noise_all[noise_idx]
    n = eeg_r.shape[0]

    train_num = round(train_per * n)
    val_num = round((n - train_num) / 2)
    test_num = n - train_num - val_num
    tr, va, te = slice(0, train_num), slice(train_num, train_num + val_num), slice(train_num + val_num, n)

    # ---- training set: combin_num independent shuffles of EEG and of noise ----------------
    def _augment(sig):
        return np.concatenate([sig[rng.permutation(sig.shape[0])] for _ in range(combin_num)], axis=0)

    eeg_train = _augment(eeg_r[tr])
    noise_train = _augment(noise_r[tr])
    snr_train_db = rng.uniform(lo, hi, eeg_train.shape[0])
    noisy_train = mix(eeg_train, noise_train, snr_train_db, snr_convention)
    noisy_train, eeg_train, _ = _standardise(noisy_train, eeg_train)

    # ---- validation / test: fixed pairing, one block per SNR level -----------------------
    levels = np.linspace(lo, hi, n_levels)

    def _blocks(eeg, noise):
        noisy = np.concatenate([mix(eeg, noise, lv, snr_convention) for lv in levels], axis=0)
        clean = np.tile(eeg, (n_levels, 1))
        return _standardise(noisy, clean)

    noisy_val, eeg_val, _ = _blocks(eeg_r[va], noise_r[va])
    noisy_test, eeg_test, test_std = _blocks(eeg_r[te], noise_r[te])

    info = {
        "seed": int(seed),
        "snr_convention": snr_convention,
        "snr_range": (lo, hi),
        "snr_levels": levels,
        "n_levels": int(n_levels),
        "noise_type": noise_type,
        "n_train_segments": int(train_num),
        "n_val_segments": int(val_num),
        "n_test_segments": int(test_num),
        "eeg_idx": eeg_idx,      # row of EEG_all used at each shuffled position
        "noise_idx": noise_idx,  # row of noise_all used at each shuffled position
    }

    if verbose:
        real = (2 * lo, 2 * hi) if snr_convention == "repo" else (lo, hi)
        print("segments  train/val/test: %d / %d / %d  (seed %d)" % (train_num, val_num, test_num, seed))
        print("SNR convention '%s': nominal %.1f..%.1f dB  ->  real (20*log10) %.1f..%.1f dB"
              % (snr_convention, lo, hi, real[0], real[1]))
        print("training data prepared", noisy_train.shape, eeg_train.shape)
        print("validation data prepared", noisy_val.shape, eeg_val.shape)
        print("test data prepared", noisy_test.shape, eeg_test.shape)

    f32 = lambda a: a.astype(np.float32)
    return (f32(noisy_train), f32(eeg_train), f32(noisy_val), f32(eeg_val),
            f32(noisy_test), f32(eeg_test), test_std, info)


def save_split_info(info, path):
    """Save the seed and the segment permutations so the exact split can be reproduced/audited."""
    np.savez(path,
             seed=info["seed"], snr_convention=info["snr_convention"],
             snr_range=np.array(info["snr_range"]), snr_levels=info["snr_levels"],
             eeg_idx=info["eeg_idx"], noise_idx=info["noise_idx"],
             n_train=info["n_train_segments"], n_val=info["n_val_segments"], n_test=info["n_test_segments"])
