"""CS877 Deliverable 1 - evidence script (data + code runs + baseline metric).
Run in the VS Code terminal and screenshot the output.
Usage:  python deliverable_evidence.py [data_dir] [nn_output_dir]
"""
import sys, os, platform, datetime
import numpy as np

DATA = r"D:/People/hadiacode/New folder/EEGdenoiseNet-master/EEGdenoiseNet-master/data/"
OUT = r"D:/People/hadiacode/New folder/EEGdenoiseNet-master/EEGdenoiseNet-master/results/EMG_unet112dense_10_rmsp_test/1/nn_output/"
if len(sys.argv) > 1: DATA = sys.argv[1]
if len(sys.argv) > 2: OUT = sys.argv[2]

line = "=" * 78
print(line)
print("CS877 Project Deliverable 1 - evidence run")
print("Run date/time :", datetime.datetime.now().strftime("%A %d %B %Y, %H:%M:%S"))
print("Python / NumPy:", platform.python_version(), "/", np.__version__)
print(line)

print("\n[1] DATA SOURCE")
print("Dataset : EEGdenoiseNet (Zhang et al., J. Neural Eng. 18, 056057, 2021)")
print("Obtained: downloaded as ZIP from https://github.com/ncclabsustech/EEGdenoiseNet")
print("Origin  : clean EEG from Cho et al. 2017 (GigaDB 100295); EOG from Kanoga 2016,")
print("          Naeem 2006, Schlogl 2007 (BCI Competition IV); 2 s epochs at 256 Hz")
print("Folder  :", DATA)

print("\n[2] LOAD DATA")
eeg = np.load(os.path.join(DATA, "EEG_all_epochs.npy"))
eog = np.load(os.path.join(DATA, "EOG_all_epochs.npy"))
for name, a in (("EEG_all_epochs.npy (clean EEG = TARGET)", eeg), ("EOG_all_epochs.npy (ocular noise)", eog)):
    print(f"{name}: rows={a.shape[0]}, columns={a.shape[1]}, dtype={a.dtype}, NaNs={int(np.isnan(a).sum())}")

print("\n[3] TARGET RANGE (task = regression / denoising, so no class balance)")
print(f"Clean EEG target: min={eeg.min():.4f}  max={eeg.max():.4f}  mean={eeg.mean():.4f}  std={eeg.std():.4f}")
print("Note: these are raw amplitudes; prepare_data() rescales each noisy/clean pair by the noisy std before training.")

print("\n[4] FIRST FIVE ROWS of EEG_all_epochs (first 8 of 512 columns shown)")
np.set_printoptions(precision=4, suppress=True, linewidth=120)
print(eeg[:5, :8])

print("\n[5] TRAIN / VALIDATION / TEST SPLIT (80/10/10 of the 3,400 epochs used)")
n = eog.shape[0]
tr = round(0.8 * n); va = round((n - tr) / 2); te = n - tr - va
print(f"epochs: train={tr}, validation={va}, test={te}")
print(f"noisy pairs after mixing x10: train={tr*10}, validation={va*10}, test={te*10}")

print("\n[6] BASELINE METRICS ON MY DATA (Simple CNN, ocular artifacts, 50 epochs)")
try:
    clean = np.load(os.path.join(OUT, "EEG_test.npy"))
    den = np.load(os.path.join(OUT, "Denoiseoutput_test.npy")).reshape(clean.shape)
    rms = lambda x: np.sqrt(np.mean(x ** 2, axis=-1))
    rrmse_t = (rms(den - clean) / rms(clean)).mean()
    a = den - den.mean(1, keepdims=True); b = clean - clean.mean(1, keepdims=True)
    cc = ((a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))).mean()
    print(f"Test segments: {clean.shape[0]}")
    print(f"Correlation coefficient (CC)  = {cc:.3f}   (paper, Simple CNN, ocular: 0.916)")
    print(f"RRMSE temporal                = {rrmse_t:.3f}   (paper, Simple CNN, ocular: 0.359)")
except FileNotFoundError:
    print("nn_output not found - check the OUT path at the top of the script.")

print("\nFinished without errors at", datetime.datetime.now().strftime("%H:%M:%S"))
print(line)