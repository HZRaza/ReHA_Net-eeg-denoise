"""Self-checks for data_prepare.py and eeg_utils.py (numpy only, no data files, ~10 s).

Run from the benchmark_networks folder:   python tests/test_data_and_metrics.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_prepare import prepare_data  # noqa: E402
from eeg_utils import (EarlyStopper, cc, epoch_permutation, measured_snr_db, psd,  # noqa: E402
                       rrmse_spectral, rrmse_temporal, summarize)

T = 512
rs = np.random.RandomState(123)
# fake dataset the same size as EEGdenoiseNet; column 0 carries a unique id per segment
EEG = rs.randn(4514, T)
EEG[:, 0] = np.arange(4514)
NOISE = rs.randn(3400, T) * 3.0
NOISE[:, 0] = np.arange(3400)


def run(**kw):
    return prepare_data(EEG, NOISE, combin_num=10, train_per=0.8, noise_type="EOG", verbose=False, **kw)


def test_shapes_and_split():
    a = run(seed=1)
    nt, et, nv, ev, ne, ee, std, info = a
    assert nt.shape == (27200, T) and et.shape == (27200, T)
    assert nv.shape == (3400, T) and ne.shape == (3400, T) and len(std) == 3400
    assert (info["n_train_segments"], info["n_val_segments"], info["n_test_segments"]) == (2720, 340, 340)
    assert nt.dtype == np.float32
    print("ok  shapes / 2720-340-340 split")


def test_no_leakage():
    info = run(seed=5)[-1]
    n_tr, n_va = info["n_train_segments"], info["n_val_segments"]
    for name, idx in (("EEG", info["eeg_idx"]), ("noise", info["noise_idx"])):
        tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te)), name
    print("ok  train / val / test use disjoint EEG and noise segments")


def test_deterministic_and_seed_changes_split():
    a, b, c = run(seed=7), run(seed=7), run(seed=8)
    assert all(np.array_equal(x, y) for x, y in zip(a[:7], b[:7]))
    assert not np.array_equal(a[-1]["eeg_idx"], c[-1]["eeg_idx"])
    print("ok  same seed -> identical data, different seed -> different split")


def test_snr_conventions():
    for conv in ("repo", "standard"):
        nt, et, nv, ev, ne, ee, std, info = run(seed=3, snr_convention=conv)
        n = 340
        for i, lv in enumerate(info["snr_levels"]):
            blk = slice(i * n, (i + 1) * n)
            got = measured_snr_db(ee[blk], ne[blk], conv).mean()
            assert abs(got - lv) < 1e-4, (conv, lv, got)
        # and the other definition differs by exactly a factor of 2
        other = "standard" if conv == "repo" else "repo"
        lv = info["snr_levels"][0]
        got = measured_snr_db(ee[:n], ne[:n], other).mean()
        assert abs(got - (2 * lv if conv == "repo" else lv / 2)) < 1e-4
    print("ok  both SNR conventions hit their nominal dB exactly (repo = 2x the standard dB)")


def test_standardisation_and_blocks():
    nt, et, nv, ev, ne, ee, std, info = run(seed=2)
    assert np.allclose(np.std(nt, axis=1), 1, atol=1e-4) and np.allclose(np.std(ne, axis=1), 1, atol=1e-4)
    # val/test: ten blocks, each block holds the same 340 clean segments (after undoing the per-row std)
    ee64 = ee.astype(np.float64) * std[:, None]
    assert np.allclose(ee64[:340], ee64[340:680], atol=1e-3)
    print("ok  unit std per segment, val/test laid out as 10 SNR blocks of the same segments")


def test_metrics():
    x = rs.randn(50, T)
    assert np.allclose(cc(x, x), 1) and np.allclose(rrmse_temporal(x, x), 0) and np.allclose(rrmse_spectral(x, x, 256), 0)
    assert np.allclose(cc(2 * x + 1, x), 1)
    assert np.allclose(rrmse_temporal(np.zeros_like(x), x), 1)
    assert psd(x, 256).shape[-1] == 241  # 0..120 Hz at 0.5 Hz resolution
    s = summarize(x + 0.1 * rs.randn(*x.shape), x, fs=256, n_levels=10)
    assert len(s["per_snr"]) == 10 and s["overall"]["cc"] > 0.99
    print("ok  metrics")


def test_early_stopper_and_shuffle():
    es = EarlyStopper(patience=3)
    seq = [1.0, 0.9, 0.95, 0.96, 0.97]
    res = [es.update(v, i + 1) for i, v in enumerate(seq)]
    assert [r[0] for r in res] == [True, True, False, False, False]
    assert [r[1] for r in res] == [False, False, False, False, True]
    assert es.best_epoch == 2 and es.best == 0.9
    assert not any(EarlyStopper(patience=0).update(v, i)[1] for i, v in enumerate([1, 2, 3, 4, 5, 6]))
    assert np.array_equal(epoch_permutation(0, 3, 100), epoch_permutation(0, 3, 100))
    assert not np.array_equal(epoch_permutation(0, 3, 100), epoch_permutation(0, 4, 100))
    print("ok  early stopping + reproducible per-epoch shuffle")


def test_reha_metrics_and_lr():
    from eeg_utils import psnr, snr, rrmse_t, rrmse_f, rrmse_s, metric_means, get_lr
    r = np.random.RandomState(5)
    x = r.randn(20, 512)
    assert np.allclose(rrmse_t(x, x), 0) and np.allclose(rrmse_f(x, x), 0) and np.allclose(rrmse_s(x, x), 0)
    assert np.all(psnr(x, x) > 150) and np.all(snr(x, x) > 60)
    e = 0.1 * r.randn(20, 512)
    assert abs(np.mean(snr(x + e, x)) - 20) < 1.0            # noise 10x smaller in amplitude -> ~20 dB
    assert abs(np.mean(rrmse_t(x + e, x)) - 0.1) < 0.02
    assert abs(np.mean(rrmse_s(2 * x, x))) < 1e-6             # amplitude ignored
    m = metric_means(x + e, x)
    assert abs(m["rrmse_total"] - (m["rrmse_t"] + m["rrmse_f"] + m["rrmse_s"]) / 3) < 1e-12
    assert get_lr("constant", 5, 40, 1e-3) == 1e-3 and get_lr("step", 10, 40, 1e-3) == 5e-4
    assert abs(get_lr("cosine_restarts", 10, 40, 1e-3) - 1e-3) < 1e-12   # restart at epoch 10
    assert get_lr("cosine", 39, 40, 1e-3) < 2e-5
    print("ok  reha metrics (PSNR/SNR/RRMSE t,f,s) + lr schedules")


if __name__ == "__main__":
    for fn in (test_shapes_and_split, test_no_leakage, test_deterministic_and_seed_changes_split,
               test_snr_conventions, test_standardisation_and_blocks, test_metrics, test_early_stopper_and_shuffle, test_reha_metrics_and_lr):
        fn()
    print("\nall checks passed")
