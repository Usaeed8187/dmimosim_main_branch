"""Frequency-grid wrapper for a memoryless Rapp power amplifier."""

import tensorflow as tf


SUPPORTED_PA_MODEL_VERSION = "rapp_v1"


def apply_rapp_pa_frequency_grid(
    x_freq,
    *,
    ibo_db,
    rho=3.0,
    model_version=SUPPORTED_PA_MODEL_VERSION,
):
    """Apply a Rapp PA independently to every physical transmit chain.

    The last two dimensions are the OFDM-symbol and FFT-bin dimensions. A
    single gain per transmit chain restores the pre-PA average slot power, so
    an IBO sweep changes nonlinear distortion without changing average SNR.
    """

    if str(model_version) != SUPPORTED_PA_MODEL_VERSION:
        raise ValueError(
            f"Unsupported PA model version {model_version!r}; "
            f"expected {SUPPORTED_PA_MODEL_VERSION!r}."
        )
    if float(rho) <= 0.0:
        raise ValueError("Rapp smoothing parameter rho must be positive.")

    x_freq = tf.convert_to_tensor(x_freq)
    if not x_freq.dtype.is_complex:
        raise TypeError("Rapp PA input must be a complex frequency grid.")

    real_dtype = x_freq.dtype.real_dtype
    ibo_db = tf.cast(ibo_db, real_dtype)
    rho = tf.cast(rho, real_dtype)
    ten = tf.cast(10.0, real_dtype)
    one = tf.cast(1.0, real_dtype)
    two = tf.cast(2.0, real_dtype)
    eps = tf.cast(tf.keras.backend.epsilon(), real_dtype)

    x_time = tf.signal.ifft(x_freq)
    input_power = tf.reduce_mean(
        tf.math.square(tf.abs(x_time)), axis=(-2, -1), keepdims=True
    )
    safe_input_power = tf.maximum(input_power, eps)
    saturation_amplitude = tf.sqrt(
        safe_input_power * tf.pow(ten, ibo_db / ten)
    )

    amplitude_ratio = tf.math.divide_no_nan(
        tf.abs(x_time), saturation_amplitude
    )
    denominator = tf.pow(
        one + tf.pow(amplitude_ratio, two * rho),
        tf.math.reciprocal(two * rho),
    )
    raw_output = tf.math.divide_no_nan(
        x_time, tf.cast(denominator, x_time.dtype)
    )

    raw_output_power = tf.reduce_mean(
        tf.math.square(tf.abs(raw_output)), axis=(-2, -1), keepdims=True
    )
    output_gain = tf.sqrt(
        tf.math.divide_no_nan(input_power, tf.maximum(raw_output_power, eps))
    )
    output_time = raw_output * tf.cast(output_gain, raw_output.dtype)
    output_time = tf.where(
        input_power > eps,
        output_time,
        tf.zeros_like(output_time),
    )
    return tf.signal.fft(output_time)


def pa_filename_token(value):
    """Return a filesystem-safe, stable numeric token."""

    return format(float(value), "g").replace("-", "m").replace(".", "p")


def pa_cache_suffix(enabled, ibo_db, rho, model_version):
    """Return the cache/result suffix for an enabled PA."""

    if not enabled:
        return ""
    return (
        f"_pa_{model_version}"
        f"_ibo_db_{pa_filename_token(ibo_db)}"
        f"_rho_{pa_filename_token(rho)}"
    )
