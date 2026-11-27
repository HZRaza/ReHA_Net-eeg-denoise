"""ReHA-Net, paper-faithful rebuild (Keras 3 / TensorFlow 2.x).  Francis & Vadivu 2026, Sci. Rep. 16:178.

What the paper states (and this file follows):
  * U-Net: 3 encoder stages, each = MSC block -> ReVIN -> Hybrid attention -> MaxPool(2); filters double per stage;
    one deep MSC bottleneck; decoder = transposed conv (x2) -> skip concat -> MSC -> ReVIN -> Hybrid attention,
    filters halve per stage; 1x1 conv output with one channel.
  * MSC block: three parallel depthwise-separable Conv1D (k = 3, 5, 7), each BN + ReLU; concatenate; 1x1 conv;
    BN + ReLU + dropout.  Dilation is mentioned but no rates are given.  No residual connection.
  * Hybrid attention, three branches, all multiply the input features:
      temporal  : global avg-pool and max-pool over time -> shared 2-layer 1x1-conv MLP (reduction r) -> sum -> sigmoid
                  (squeeze-and-excitation style, one weight per feature channel)
      spatial   : mean and max over channels -> concat -> Conv1D -> sigmoid  (one weight per time step)
      frequency : rfft along time -> magnitude -> Conv1D along frequency -> sigmoid -> interpolate to the time axis
    The three outputs are concatenated on the channel axis, then 1x1 conv + dropout.
  * ReVIN at the input and after every MSC block: per-sample, per-channel normalisation over time + learned affine,
    no explicit reversal.
  * Loss: MSE + CC loss + PSD (FFT-magnitude MAE) loss, weights "empirically set, MSE emphasised" (see make_loss).

What the paper does NOT give, so these are my choices and are exposed as options (each one is an ablation arm):
  dropout rate 0.2, MSC dilation rates (default 1,1,1), attention reduction r = 8, spatial conv kernel 7,
  frequency conv kernel 5, base width 46 (chosen so the trainable parameters come to ~2.15 M, the paper's ~2.1 M), whether attention/MSC have a residual
  (default off, as described), fusion 1x1-conv width = input channels.
"""
import tensorflow as tf
from tensorflow.keras import layers

from model_reha import InstanceNorm, InstanceDenorm, make_loss  # noqa: F401  (re-exported)

K = tf.keras


def msc_block(x, filters, dilations, dropout, residual, name):
    branches = []
    for k, d in zip((3, 5, 7), dilations):
        b = layers.SeparableConv1D(filters, k, dilation_rate=d, padding="same", name="%s_k%d" % (name, k))(x)
        b = layers.BatchNormalization(name="%s_bn%d" % (name, k))(b)
        branches.append(layers.Activation("relu", name="%s_relu%d" % (name, k))(b))
    y = layers.Concatenate(name=name + "_cat")(branches)
    y = layers.Conv1D(filters, 1, name=name + "_fuse")(y)
    y = layers.BatchNormalization(name=name + "_fuse_bn")(y)
    y = layers.Activation("relu", name=name + "_fuse_relu")(y)
    if dropout > 0:
        y = layers.Dropout(dropout, name=name + "_drop")(y)
    if residual:
        skip = x if x.shape[-1] == filters else layers.Conv1D(filters, 1, name=name + "_skip")(x)
        y = layers.Add(name=name + "_add")([y, skip])
    return y


class HybridAttention(layers.Layer):
    """Temporal (SE-style channel gate) + spatial (per-time-step gate) + frequency gate, concat -> 1x1 conv -> dropout."""

    def __init__(self, channels, reduction=8, spatial_kernel=7, freq_kernel=5, dropout=0.2,
                 temporal=True, spatial=True, frequency=True, residual=False, **kw):
        super().__init__(**kw)
        self.channels, self.reduction = channels, reduction
        self.spatial_kernel, self.freq_kernel, self.drop_rate = spatial_kernel, freq_kernel, dropout
        self.use_t, self.use_s, self.use_f, self.residual = temporal, spatial, frequency, residual
        hidden = max(channels // reduction, 2)
        if temporal:
            self.mlp1 = layers.Conv1D(hidden, 1, activation="relu", name="t_mlp1")
            self.mlp2 = layers.Conv1D(channels, 1, name="t_mlp2")
        if spatial:
            self.s_conv = layers.Conv1D(1, spatial_kernel, padding="same", activation="sigmoid", name="s_gate")
        if frequency:
            self.f_conv = layers.Conv1D(channels, freq_kernel, padding="same", activation="sigmoid", name="f_gate")
        n_branches = int(temporal) + int(spatial) + int(frequency)
        self.n_branches = n_branches
        self.fuse = layers.Conv1D(channels, 1, name="fuse") if n_branches > 0 else None
        self.drop = layers.Dropout(dropout) if dropout > 0 else None

    def build(self, shape):
        self.T = int(shape[1])
        c = self.channels
        if self.use_t:
            self.mlp1.build((None, 1, c))
            self.mlp2.build((None, 1, self.mlp1.filters))
        if self.use_s:
            self.s_conv.build((None, self.T, 2))
        if self.use_f:
            self.f_conv.build((None, self.T // 2 + 1, c))
        self.fuse.build((None, self.T, c * self.n_branches))
        super().build(shape)

    def call(self, x, training=None):
        outs = []
        if self.use_t:
            avg = tf.reduce_mean(x, axis=1, keepdims=True)                       # (B,1,C)
            mx = tf.reduce_max(x, axis=1, keepdims=True)
            w = tf.sigmoid(self.mlp2(self.mlp1(avg)) + self.mlp2(self.mlp1(mx)))  # shared MLP
            outs.append(x * w)
        if self.use_s:
            s = tf.concat([tf.reduce_mean(x, -1, keepdims=True), tf.reduce_max(x, -1, keepdims=True)], -1)
            outs.append(x * self.s_conv(s))                                       # (B,T,1) gate
        if self.use_f:
            spec = tf.signal.rfft(tf.transpose(x, [0, 2, 1]))                     # (B,C,F) complex
            mag = tf.abs(spec) / tf.sqrt(tf.cast(self.T, tf.float32))             # (B,C,F)
            g = self.f_conv(tf.transpose(mag, [0, 2, 1]))                          # (B,F,C) in (0,1)
            g = tf.image.resize(g[:, :, None, :], (self.T, 1), method="bilinear")[:, :, 0, :]   # (B,T,C)
            outs.append(x * g)
        y = outs[0] if len(outs) == 1 else tf.concat(outs, -1)
        y = self.fuse(y)
        if self.drop is not None:
            y = self.drop(y, training=training)
        return x + y if self.residual else y

    def get_config(self):
        return {**super().get_config(), "channels": self.channels, "reduction": self.reduction,
                "spatial_kernel": self.spatial_kernel, "freq_kernel": self.freq_kernel, "dropout": self.drop_rate,
                "temporal": self.use_t, "spatial": self.use_s, "frequency": self.use_f, "residual": self.residual}


DEFAULTS = dict(base=46, dropout=0.2, dilations=(1, 1, 1), reduction=8, revin="norm",
                temporal_att=True, spatial_att=True, freq_att=True, msc_residual=False, att_residual=False)


def build_reha_net_pf(length=512, **cfg):
    """revin: 'none' | 'norm' (paper: input + after every MSC) | 'input' (input only) | 'reverse' (norm + real reversal)."""
    c = {**DEFAULTS, **cfg}
    base, drop, dil = c["base"], c["dropout"], tuple(c["dilations"])
    use_att = c["temporal_att"] or c["spatial_att"] or c["freq_att"]
    every = c["revin"] in ("norm", "reverse")
    ch = [base, 2 * base, 4 * base, 8 * base]

    def revin(x, name):
        return InstanceNorm(name=name)(x) if every else x

    def att(x, filters, name):
        if not use_att:
            return x
        return HybridAttention(filters, c["reduction"], dropout=drop, temporal=c["temporal_att"],
                               spatial=c["spatial_att"], frequency=c["freq_att"], residual=c["att_residual"],
                               name=name)(x)

    inp = layers.Input((length, 1), name="noisy_eeg")
    x = inp
    if c["revin"] in ("norm", "reverse", "input"):
        x = InstanceNorm(name="revin_in")(x)

    skips = []
    for i in range(3):
        x = msc_block(x, ch[i], dil, drop, c["msc_residual"], "enc%d_msc" % (i + 1))
        x = revin(x, "enc%d_revin" % (i + 1))
        x = att(x, ch[i], "enc%d_att" % (i + 1))
        skips.append(x)
        x = layers.MaxPooling1D(2, name="pool%d" % (i + 1))(x)
    x = msc_block(x, ch[3], dil, drop, c["msc_residual"], "bottleneck_msc")
    x = revin(x, "bottleneck_revin")
    for i in (2, 1, 0):
        x = layers.Conv1DTranspose(ch[i], 2, strides=2, padding="same", name="up%d" % (i + 1))(x)
        x = layers.Concatenate(name="skip%d" % (i + 1))([x, skips[i]])
        x = msc_block(x, ch[i], dil, drop, c["msc_residual"], "dec%d_msc" % (i + 1))
        x = revin(x, "dec%d_revin" % (i + 1))
        x = att(x, ch[i], "dec%d_att" % (i + 1))
    out = layers.Conv1D(1, 1, name="head")(x)
    if c["revin"] == "reverse":
        out = InstanceDenorm(name="revin_reverse")([out, inp])
    return K.Model(inp, out, name="ReHA_Net_PF")
