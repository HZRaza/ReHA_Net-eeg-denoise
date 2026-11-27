"""ReHA-Net rebuilt in Keras (TensorFlow 2.x / Keras 3) from the description in Francis & Vadivu 2026.

The authors' code is not public, so details the paper does not pin down are my reading of it:
  * U-Net, 3 encoder stages + bottleneck + 3 decoder stages (strided-conv down, upsample+conv up)
  * MSC block: three parallel depthwise-separable convs (kernels 3/5/7) -> concat -> 1x1 fuse, residual
  * Hybrid attention = temporal gate + frequency gate (+ residual). The paper's "spatial" attention is a
    pass-through for a single EEG channel, so it is left out here.
      temporal : sigmoid(Conv1D(7)([mean_c, max_c]))  over time, multiplies the features
      frequency: rfft along time, sigmoid(Conv1D(5)([mean_c, max_c] of log1p|X|)) over frequency
                 multiplies the spectrum, irfft back
  * "ReVIN": per-instance normalisation + learned affine at the input. The paper has no explicit
    reversal; revin='reverse' adds the real reversal at the output (an improvement option).
  * loss: MSE + lambda_cc * (1 - Pearson CC) + lambda_psd * MAE(|rfft| / sqrt(T))
Default size (base=36, 2 MSC blocks per stage) is about 2.1 M parameters; the real count is printed.
"""
import tensorflow as tf
from tensorflow.keras import layers

K = tf.keras


class InstanceNorm(layers.Layer):
    """x -> (x - mean_t) / (std_t + eps) * gamma + beta, statistics per sample and channel over time."""

    def __init__(self, eps=1e-5, affine=True, **kw):
        super().__init__(**kw)
        self.eps, self.affine = eps, affine

    def build(self, shape):
        c = shape[-1]
        if self.affine:
            self.gamma = self.add_weight(name="gamma", shape=(c,), initializer="ones")
            self.beta = self.add_weight(name="beta", shape=(c,), initializer="zeros")

    def call(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        std = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True) + self.eps)
        y = (x - mean) / std
        return y * self.gamma + self.beta if self.affine else y

    def get_config(self):
        return {**super().get_config(), "eps": self.eps, "affine": self.affine}


class InstanceDenorm(layers.Layer):
    """Real reversal: y_out = y * std(x_in) + mean(x_in). inputs = [y, x_in]."""

    def __init__(self, eps=1e-5, **kw):
        super().__init__(**kw)
        self.eps = eps

    def call(self, inputs):
        y, x = inputs
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        std = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True) + self.eps)
        return y * std + mean

    def get_config(self):
        return {**super().get_config(), "eps": self.eps}


class HybridAttention(layers.Layer):
    """Temporal + frequency attention with a residual connection. Input (B, T, C), T must be static."""

    def __init__(self, temporal=True, frequency=True, **kw):
        super().__init__(**kw)
        self.temporal, self.frequency = temporal, frequency
        if temporal:
            self.t_conv = layers.Conv1D(1, 7, padding="same", activation="sigmoid", name="t_gate")
        if frequency:
            self.f_conv = layers.Conv1D(1, 5, padding="same", activation="sigmoid", name="f_gate")

    def build(self, shape):
        self.T = int(shape[1])
        if self.temporal:
            self.t_conv.build((None, self.T, 2))
        if self.frequency:
            self.f_conv.build((None, self.T // 2 + 1, 2))
        super().build(shape)

    def call(self, x):
        y = x
        if self.temporal:
            s = tf.concat([tf.reduce_mean(y, -1, keepdims=True), tf.reduce_max(y, -1, keepdims=True)], -1)
            y = y * self.t_conv(s)
        if self.frequency:
            spec = tf.signal.rfft(tf.transpose(y, [0, 2, 1]))                    # (B, C, F) complex
            mag = tf.math.log1p(tf.abs(spec))                                    # (B, C, F)
            s = tf.stack([tf.reduce_mean(mag, 1), tf.reduce_max(mag, 1)], -1)    # (B, F, 2)
            w = tf.transpose(self.f_conv(s), [0, 2, 1])                          # (B, 1, F)
            # scale real and imaginary parts by the real gate (same maths as spec * w, but a
            # float -> complex cast would log a harmless "discards the imaginary part" warning in the gradient)
            spec = tf.complex(tf.math.real(spec) * w, tf.math.imag(spec) * w)
            y = tf.transpose(tf.signal.irfft(spec, fft_length=[self.T]), [0, 2, 1])
        return x + y

    def get_config(self):
        return {**super().get_config(), "temporal": self.temporal, "frequency": self.frequency}


def msc_block(x, filters, name):
    """Multi-scale conv: depthwise-separable k=3,5,7 in parallel -> concat -> 1x1 fuse, residual."""
    branches = []
    for k in (3, 5, 7):
        b = layers.SeparableConv1D(filters, k, padding="same", name="%s_k%d" % (name, k))(x)
        b = layers.BatchNormalization(name="%s_bn%d" % (name, k))(b)
        branches.append(layers.Activation("relu")(b))
    y = layers.Concatenate(name=name + "_cat")(branches)
    y = layers.Conv1D(filters, 1, name=name + "_fuse")(y)
    y = layers.BatchNormalization(name=name + "_fuse_bn")(y)
    skip = x if x.shape[-1] == filters else layers.Conv1D(filters, 1, name=name + "_skip")(x)
    return layers.Activation("relu")(layers.Add(name=name + "_add")([y, skip]))


def _stage(x, filters, blocks, attn, name):
    for j in range(blocks):
        x = msc_block(x, filters, "%s_msc%d" % (name, j))
    if attn is not None:
        x = HybridAttention(attn[0], attn[1], name=name + "_att")(x)
    return x


def build_reha_net(length=512, base=36, blocks=2, revin="norm", temporal_att=True, freq_att=True):
    """revin: 'none' | 'norm' (instance norm + affine, as in the paper) | 'reverse' (norm + real reversal)."""
    attn = (temporal_att, freq_att) if (temporal_att or freq_att) else None
    ch = [base, 2 * base, 4 * base, 8 * base]
    inp = layers.Input((length, 1), name="noisy_eeg")
    x = inp
    if revin in ("norm", "reverse"):
        x = InstanceNorm(name="revin")(x)
    x = layers.Conv1D(base, 7, padding="same", name="stem")(x)

    skips = []
    for i in range(3):
        x = _stage(x, ch[i], blocks, attn, "enc%d" % (i + 1))
        skips.append(x)
        x = layers.Conv1D(ch[i + 1], 3, strides=2, padding="same", name="down%d" % (i + 1))(x)
    x = _stage(x, ch[3], blocks, attn, "bottleneck")
    for i in (2, 1, 0):
        x = layers.UpSampling1D(2, name="up%d" % (i + 1))(x)
        x = layers.Conv1D(ch[i], 3, padding="same", name="upconv%d" % (i + 1))(x)
        x = layers.Concatenate(name="skip%d" % (i + 1))([x, skips[i]])
        x = _stage(x, ch[i], blocks, attn, "dec%d" % (i + 1))
    out = layers.Conv1D(1, 1, name="head")(x)
    if revin == "reverse":
        out = InstanceDenorm(name="revin_reverse")([out, inp])
    return K.Model(inp, out, name="ReHA_Net")


BAND_EDGES = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 40))   # delta, theta, alpha, beta, gamma (Hz)


def make_loss(lambda_cc=0.05, lambda_psd=0.05, lambda_band=0.0, fs=256, band_weights=None):
    """Composite loss: MSE + lambda_cc*(1-CC) + lambda_psd*MAE(|rfft|/sqrt(T)) + lambda_band * band-power term.
    y, pred: (B, T[,1]).
    Band-power term (the Phase 3 proposal): for each EEG band b, d_b = ln((P_pred,b + eps)/(P_clean,b + eps)) with
    P = sum of |rfft|^2 / T inside the band, and the term is mean_b(w_b * d_b^2).  It is zero when every band's power
    matches the clean signal, so it directly penalises the systematic under-power that MSE tolerates.
    band_weights: optional length-5 weights (default all 1)."""
    import numpy as np
    bw = np.ones(len(BAND_EDGES), np.float32) if band_weights is None else np.asarray(band_weights, np.float32)

    def loss_fn(y, pred):
        pred = tf.reshape(pred, tf.shape(y))
        mse = tf.reduce_mean(tf.square(pred - y))
        total = mse
        if lambda_cc:
            pc, yc = pred - tf.reduce_mean(pred, 1, keepdims=True), y - tf.reduce_mean(y, 1, keepdims=True)
            cc = tf.reduce_sum(pc * yc, 1) / (tf.sqrt(tf.reduce_sum(pc ** 2, 1) * tf.reduce_sum(yc ** 2, 1)) + 1e-8)
            total += lambda_cc * (1.0 - tf.reduce_mean(cc))
        if lambda_psd:
            t = tf.cast(tf.shape(y)[1], tf.float32)
            fp, fy = tf.abs(tf.signal.rfft(pred)), tf.abs(tf.signal.rfft(y))
            total += lambda_psd * tf.reduce_mean(tf.abs(fp - fy)) / tf.sqrt(t)
        if lambda_band:
            T = int(y.shape[1])
            f = np.fft.rfftfreq(T, 1.0 / fs)
            masks = tf.constant(np.stack([((f >= lo) & (f < hi)) for lo, hi in BAND_EDGES], 1).astype(np.float32))  # (F, 5)
            ep = tf.square(tf.abs(tf.signal.rfft(pred))) / T
            ey = tf.square(tf.abs(tf.signal.rfft(y))) / T
            bp, by = tf.matmul(ep, masks), tf.matmul(ey, masks)                                   # (B, 5)
            d = tf.math.log((bp + 1e-4) / (by + 1e-4))
            total += lambda_band * tf.reduce_mean(tf.constant(bw) * tf.square(d))
        return total

    return loss_fn
