import unittest

import numpy as np
import tensorflow as tf

from dmimo.channel.dmimo_channels import dMIMOChannels
from dmimo.channel.pa_nonlinearity import (
    apply_rapp_pa_frequency_grid,
    pa_cache_suffix,
)
from dmimo.config import Ns3Config


class RappPaFrequencyGridTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        samples = rng.normal(size=(2, 1, 3, 4, 64)) + 1j * rng.normal(
            size=(2, 1, 3, 4, 64)
        )
        self.x = tf.constant(samples.astype(np.complex64))

    @staticmethod
    def _chain_power(x):
        x_time = tf.signal.ifft(x)
        return tf.reduce_mean(tf.abs(x_time) ** 2, axis=(-2, -1))

    def test_preserves_shape_dtype_and_average_chain_power(self):
        actual = apply_rapp_pa_frequency_grid(self.x, ibo_db=3.0, rho=3.0)

        self.assertEqual(actual.shape, self.x.shape)
        self.assertEqual(actual.dtype, self.x.dtype)
        np.testing.assert_allclose(
            self._chain_power(actual).numpy(),
            self._chain_power(self.x).numpy(),
            rtol=2e-6,
            atol=2e-7,
        )

    def test_distortion_increases_as_ibo_decreases(self):
        high = apply_rapp_pa_frequency_grid(self.x, ibo_db=30.0, rho=3.0)
        low = apply_rapp_pa_frequency_grid(self.x, ibo_db=0.0, rho=3.0)
        high_evm = tf.reduce_mean(tf.abs(high - self.x) ** 2)
        low_evm = tf.reduce_mean(tf.abs(low - self.x) ** 2)

        self.assertLess(float(high_evm), float(low_evm))
        self.assertLess(float(high_evm), 1e-6)

    def test_zero_chain_stays_zero(self):
        zeros = tf.zeros((1, 1, 2, 3, 32), dtype=tf.complex64)
        actual = apply_rapp_pa_frequency_grid(zeros, ibo_db=0.0, rho=3.0)
        np.testing.assert_array_equal(actual.numpy(), zeros.numpy())

    def test_cache_suffix_is_parameter_specific(self):
        self.assertEqual(pa_cache_suffix(False, 5.0, 3.0, "rapp_v1"), "")
        self.assertEqual(
            pa_cache_suffix(True, 6.5, 3.0, "rapp_v1"),
            "_pa_rapp_v1_ibo_db_6p5_rho_3",
        )

    def test_dmimo_channel_applies_pa_before_frequency_channel(self):
        cfg = Ns3Config(pa_enabled=True, pa_ibo_db=0.0, pa_rho=3.0)
        channel = dMIMOChannels(
            cfg,
            "dMIMO",
            add_noise=False,
            return_channel=True,
        )
        h_freq = tf.ones((2, 1, 1, 1, 3, 4, 64), dtype=tf.complex64)
        rx_snr = tf.zeros((2, 1, 1, 4), dtype=tf.float32)
        rx_power = tf.zeros((2, 1), dtype=tf.float32)
        channel._load_channel = lambda *args, **kwargs: (
            h_freq,
            rx_snr,
            rx_power,
        )

        actual, returned_h = channel([self.x, 5])
        expected_x = apply_rapp_pa_frequency_grid(
            self.x, ibo_db=0.0, rho=3.0
        )
        expected = tf.reduce_sum(expected_x, axis=(1, 2))[:, tf.newaxis, tf.newaxis]

        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-6)
        np.testing.assert_array_equal(returned_h.numpy(), h_freq.numpy())


if __name__ == "__main__":
    unittest.main()
