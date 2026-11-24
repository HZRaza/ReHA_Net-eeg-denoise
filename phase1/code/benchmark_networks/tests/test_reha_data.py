"""Tests for data_reha.py (numpy only). Run from benchmark_networks:  python tests/test_reha_data.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_reha import prepare_reha_data
from eeg_utils import measured_snr_db


def fake(n, seed, scale=1.0):
    r = np.random.RandomState(seed)
    return r.randn(n, 512) * scale * r.uniform(0.5, 2, (n, 1))


def run(noise_type, conv="standard"):
    eeg, eog, emg = fake(600, 1), fake(400, 2, 3.0), fake(700, 3, 0.2)
    return prepare_reha_data(eeg, eog, emg, noise_type=noise_type, combin_num=3, seed=5,
                             snr_convention=conv, verbose=False)


def test_shapes_and_levels():
    trn, trc, vn, vc, tn, tc, _, info = run("MIXED")
    n_tr, n_va, n_te = info["n_train_segments"], info["n_val_segments"], info["n_test_segments"]
    assert (n_tr, n_va, n_te) == (480, 60, 60)
    assert trn.shape == (3 * 480, 512) and vn.shape == (6 * 60, 512) and tn.shape == (6 * 60, 512)
    assert np.allclose(info["snr_levels"], [-5, 0, 5, 10, 15, 20])
    # same clean segments in every block; they differ only by the per-segment 1/std scaling
    r = tc[:60] / tc[60:120]
    assert np.allclose(r, r[:, :1], rtol=1e-3, atol=1e-3)


def test_measured_snr_matches_nominal():
    for conv, factor in (("standard", 1.0), ("repo", 2.0)):
        _, _, _, _, tn, tc, _, info = run("MIXED", conv)
        m = 60
        for i, lv in enumerate(info["snr_levels"]):
            sl = slice(i * m, (i + 1) * m)
            # noisy and clean are both divided by the noisy std, so the ratio is unchanged
            got = measured_snr_db(tc[sl], tn[sl], "standard").mean()
            assert abs(got - factor * lv) < 1.5, (conv, lv, got)


def test_no_leakage_between_splits():
    """Each noise pool and the EEG pool are split before mixing: indices must be disjoint."""
    _, _, _, _, _, _, _, info = run("MIXED")
    idx = info["eeg_idx"]
    n_tr, n_va = info["n_train_segments"], info["n_val_segments"]
    tr, va, te = set(idx[:n_tr]), set(idx[n_tr:n_tr + n_va]), set(idx[n_tr + n_va:])
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_noise_labels_and_types():
    _, _, _, _, _, _, _, info = run("MIXED")
    lab = info["test_noise_label"]
    assert set(np.unique(lab)) == {0, 1} and abs(lab.mean() - 0.5) < 0.1
    assert (lab[:60] == lab[60:120]).all()             # same noise type per segment in every block
    for nt, want in (("EOG", {0}), ("EMG", {1})):
        assert set(np.unique(run(nt)[-1]["test_noise_label"])) == want


def test_reproducible():
    a, b = run("MIXED"), run("MIXED")
    assert all(np.array_equal(x, y) for x, y in zip(a[:6], b[:6]))


def test_bandpass_runs():
    eeg, eog, emg = fake(100, 1), fake(80, 2), fake(90, 3)
    out = prepare_reha_data(eeg, eog, emg, combin_num=1, seed=1, use_bandpass=True, verbose=False)
    assert np.isfinite(out[0]).all()


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
