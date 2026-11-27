"""ReHA-Net v2 = the rescripted baseline.  v1 (the only model that trained well) + the changes the diagnosis supports.

  1. GLOBAL RESIDUAL PATH: output = input - head(features).  The network predicts the ARTIFACT, so "do nothing" is the
     easy solution for clean inputs.  Targets the measured damage at high input SNR (v1 -1.1 dB at 20 dB, pf -7.0 dB).
     The head starts at zero, so training starts from output = input.
  2. Small SpatialDropout1D in the bottleneck only (v1 overfits; pf's dropout everywhere made it underfit).
  3. ReVIN (instance norm) at the input only, as in v1; the residual path adds the real amplitude back.
Everything else (U-Net, MSC blocks with residual, temporal + frequency attention, base 36) is identical to v1.
"""
import tensorflow as tf
from tensorflow.keras import layers

from model_reha import InstanceNorm, _stage, make_loss  # noqa: F401  (make_loss re-exported)

K = tf.keras
DEFAULTS = dict(base=36, blocks=2, dropout=0.1, revin="norm", temporal_att=True, freq_att=True, residual=True)


def build_reha_net_v2(length=512, base=36, blocks=2, dropout=0.1, revin="norm", temporal_att=True, freq_att=True,
                      residual=True):
    attn = (temporal_att, freq_att) if (temporal_att or freq_att) else None
    ch = [base, 2 * base, 4 * base, 8 * base]
    inp = layers.Input((length, 1), name="noisy_eeg")
    x = InstanceNorm(name="revin")(inp) if revin != "none" else inp
    x = layers.Conv1D(base, 7, padding="same", name="stem")(x)
    skips = []
    for i in range(3):
        x = _stage(x, ch[i], blocks, attn, "enc%d" % (i + 1))
        skips.append(x)
        x = layers.Conv1D(ch[i + 1], 3, strides=2, padding="same", name="down%d" % (i + 1))(x)
    x = _stage(x, ch[3], blocks, attn, "bottleneck")
    if dropout > 0:
        x = layers.SpatialDropout1D(dropout, name="bottleneck_drop")(x)
    for i in (2, 1, 0):
        x = layers.UpSampling1D(2, name="up%d" % (i + 1))(x)
        x = layers.Conv1D(ch[i], 3, padding="same", name="upconv%d" % (i + 1))(x)
        x = layers.Concatenate(name="skip%d" % (i + 1))([x, skips[i]])
        x = _stage(x, ch[i], blocks, attn, "dec%d" % (i + 1))
    if residual:
        art = layers.Conv1D(1, 1, kernel_initializer="zeros", name="artifact_head")(x)
        out = layers.Subtract(name="clean_estimate")([inp, art])      # input - predicted artifact
    else:
        out = layers.Conv1D(1, 1, name="head")(x)
    return K.Model(inp, out, name="ReHA_Net_v2")
