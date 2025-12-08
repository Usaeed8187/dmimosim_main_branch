import tensorflow as tf
import numpy as np


def complex_pinv(a:tf.Tensor, rcond=None):
    """Compute Moore-Penrose pseudo-inverse supporting complex dtypes.

    This is a small wrapper inspired by TensorFlow's `tf.linalg.pinv` but
    handles complex-valued input by using the appropriate real `eps` value
    for default ``rcond`` and by relying on ``tf.linalg.svd`` which supports
    complex inputs.

    Args:
        a: Tensor-like, shape [..., M, N], can be complex or real.
        rcond: Optional scalar or tensor giving relative cutoff for small
            singular values. If ``None``, a default based on machine eps is used
            (similar to ``tf.linalg.pinv``).

    Returns:
        a_pinv: pseudo-inverse of ``a`` with shape [..., N, M].
    """

    a = tf.convert_to_tensor(a, name="a")

    # Validate minimal rank
    if a.shape.rank is not None and a.shape.rank < 2:
        raise ValueError("Input `a` must have at least 2 dimensions")

    # Determine dtype to use for eps if rcond is None. For complex types
    # use the underlying real dtype (e.g., complex64 -> float32).
    dtype = a.dtype
    if dtype.is_complex:
        real_dtype = dtype.real_dtype
    else:
        real_dtype = dtype

    # Default rcond (match tf.linalg.pinv default heuristic)
    if rcond is None:
        # get matrix dims (static if possible)
        shape = a.shape
        if shape[-2] is not None and shape[-1] is not None:
            max_rows_cols = float(max(shape[-2], shape[-1]))
        else:
            # dynamic fallback
            s = tf.shape(a)
            max_rows_cols = tf.cast(tf.maximum(s[-2], s[-1]), real_dtype)
        rcond = 10.0 * max_rows_cols * np.finfo(real_dtype.as_numpy_dtype).eps

    # Ensure rcond is tensor of appropriate real dtype
    rcond = tf.convert_to_tensor(rcond, dtype=real_dtype, name="rcond")

    # Compute SVD. tf.linalg.svd returns real singular values even for complex A.
    # We request full_matrices=False to match tf.linalg.pinv behaviour.
    s, u, v = tf.linalg.svd(a, full_matrices=False, compute_uv=True)

    # cutoff = rcond * max(singular_values)
    # s has shape [..., K] where K = min(M,N)
    cutoff = rcond * tf.reduce_max(s, axis=-1)

    # Saturate small singular values to inf so that 1. / s = 0.
    cutoff_exp = tf.expand_dims(cutoff, -1)
    inf = tf.constant(np.inf, dtype=s.dtype)
    s_safe = tf.where(s > cutoff_exp, s, inf)

    # Compute v @ inv(s) @ u^H  (note: v has shape [..., N, K],
    # s_safe is [..., K], u is [..., M, K])
    # Cast singular-value denominator to the same dtype as "v" so division
    # works for complex-valued "v" (s_safe is real dtype).
    denom = tf.cast(tf.expand_dims(s_safe, -2), v.dtype)

    # v_over_s = v / denom # Not used due to TF bugs with complex division when denom has inf values
    # Doing v / denom has some bugs in TensorFlow if denom has inf values and dtype is complex.
    # The following alternative implementation avoids these issues by doing absolute division
    # and subtracting the angle separately.
    angle_diff = tf.math.angle(v) - tf.math.angle(denom)
    abs_div = tf.abs(v) / tf.abs(denom)
    v_over_s = tf.cast(abs_div, v.dtype) * tf.exp(tf.complex(tf.zeros_like(angle_diff), angle_diff))
    # a_pinv = v_over_s @ conj(u)
    a_pinv = tf.matmul(v_over_s, u, adjoint_b=True)

    # Set static shape if available
    if a.shape is not None and a.shape.rank is not None:
        a_pinv.set_shape(a.shape[:-2].concatenate([a.shape[-1], a.shape[-2]]))

    return a_pinv


if __name__ == "__main__":
    # Quick smoke test comparing against tf.linalg.pinv for real and complex
    tf.get_logger().setLevel("ERROR")

    # real test
    A = tf.constant([[1.0, 0.4, 0.5], [0.4, 0.2, 0.25], [0.5, 0.25, 0.35]], dtype=tf.float32)
    p1 = complex_pinv(A)
    p2 = tf.linalg.pinv(A)
    print("real max diff:", tf.reduce_max(tf.abs(p1 - p2)).numpy())

    # complex test - compare against numpy.linalg.pinv (tf.linalg.pinv does
    # not accept complex inputs in some TF versions).
    B = tf.constant([[1 + 1j, 0.2 - 0.1j], [0.2 + 0.1j, 0.5 + 0.3j]], dtype=tf.complex64)
    p1 = complex_pinv(B)
    p2_np = np.linalg.pinv(B.numpy())
    p2 = tf.constant(p2_np, dtype=tf.complex64)
    print("complex max diff:", tf.reduce_max(tf.abs(p1 - p2)).numpy())
