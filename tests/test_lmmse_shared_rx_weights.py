import numpy as np
import tensorflow as tf
import unittest
from sionna.ofdm import LSChannelEstimator, ResourceGrid

from dmimo.channel.interpolation import LMMSELinearInterp


def _resource_grid():
    return ResourceGrid(
        num_ofdm_symbols=4,
        fft_size=16,
        subcarrier_spacing=15e3,
        num_tx=1,
        num_streams_per_tx=2,
        cyclic_prefix_length=4,
        num_guard_carriers=[0, 0],
        dc_null=False,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[1, 3],
    )


def _covariance(size):
    indices = np.arange(size)
    values = np.power(0.8, np.abs(indices[:, None] - indices[None, :]))
    return tf.constant(values.astype(np.complex64))


def _received_grid(num_rx_ant=3):
    rng = np.random.default_rng(1234)
    values = (
        rng.standard_normal((1, 1, num_rx_ant, 4, 16))
        + 1j * rng.standard_normal((1, 1, num_rx_ant, 4, 16))
    )
    return tf.constant(values.astype(np.complex64))


class SharedRxWeightsTest(unittest.TestCase):
    def test_shared_rx_weights_match_independent_solves(self):
        rg = _resource_grid()
        covariance = _covariance(rg.num_effective_subcarriers)
        received = _received_grid()
        nvar = tf.constant(0.05, tf.float32)

        reference_interpolator = LMMSELinearInterp(
            rg.pilot_pattern, covariance, share_rx_weights=False
        )
        reference_estimator = LSChannelEstimator(
            rg, interpolator=reference_interpolator
        )
        reference_h = []
        reference_err = []
        for ant in range(received.shape[2]):
            h_hat, err_var = reference_estimator(
                [received[:, :, ant : ant + 1, :, :], nvar]
            )
            reference_h.append(h_hat)
            reference_err.append(err_var)
        reference_h = tf.concat(reference_h, axis=2)
        reference_err = tf.concat(reference_err, axis=2)

        shared_interpolator = LMMSELinearInterp(
            rg.pilot_pattern, covariance, share_rx_weights=True
        )
        shared_estimator = LSChannelEstimator(rg, interpolator=shared_interpolator)
        shared_h, shared_err = shared_estimator([received, nvar])

        np.testing.assert_allclose(
            shared_h.numpy(), reference_h.numpy(), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            shared_err.numpy(), reference_err.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_shared_rx_weights_reject_receiver_specific_error_variance(self):
        rg = _resource_grid()
        interpolator = LMMSELinearInterp(
            rg.pilot_pattern,
            _covariance(rg.num_effective_subcarriers),
            share_rx_weights=True,
        )
        estimator = LSChannelEstimator(rg, interpolator=interpolator)
        receiver_specific_nvar = tf.constant([[[0.05, 0.1, 0.2]]], tf.float32)

        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError, "share_rx_weights"
        ):
            estimator([_received_grid(), receiver_specific_nvar])


if __name__ == "__main__":
    unittest.main()
