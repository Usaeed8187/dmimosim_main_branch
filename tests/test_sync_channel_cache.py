import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


# Load this module in isolation: these tests exercise cache selection and do
# not require TensorFlow/Sionna, which are optional in the lightweight test
# environment.
module_name = "dmimo.channel.rc_pred_freq_mimo"
stubbed_module_names = (
    "tensorflow",
    "dmimo",
    "dmimo.config",
    "dmimo.channel",
    "dmimo.channel.channel_estimation",
    module_name,
)
missing_module = object()
saved_modules = {
    name: sys.modules.get(name, missing_module) for name in stubbed_module_names
}

sys.modules["tensorflow"] = types.ModuleType("tensorflow")
dmimo_module = types.ModuleType("dmimo")
dmimo_module.__path__ = []
sys.modules["dmimo"] = dmimo_module
config_module = types.ModuleType("dmimo.config")
config_module.Ns3Config = object
config_module.RCConfig = object
sys.modules["dmimo.config"] = config_module
channel_module = types.ModuleType("dmimo.channel")
channel_module.__path__ = []
sys.modules["dmimo.channel"] = channel_module
estimation_module = types.ModuleType("dmimo.channel.channel_estimation")
estimation_module.lmmse_channel_estimation = None
estimation_module.get_received_pilot_symbols = None
sys.modules["dmimo.channel.channel_estimation"] = estimation_module

module_path = (
    Path(__file__).resolve().parents[1]
    / "dmimo"
    / "channel"
    / "rc_pred_freq_mimo.py"
)
spec = importlib.util.spec_from_file_location(module_name, module_path)
rc_pred_freq_mimo = importlib.util.module_from_spec(spec)
sys.modules[module_name] = rc_pred_freq_mimo
spec.loader.exec_module(rc_pred_freq_mimo)
for name, saved_module in saved_modules.items():
    if saved_module is missing_module:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = saved_module

standard_rc_pred_freq_mimo = rc_pred_freq_mimo.standard_rc_pred_freq_mimo
synchronization_cache_suffix = rc_pred_freq_mimo.synchronization_cache_suffix


class SynchronizationCacheSuffixTest(unittest.TestCase):
    def test_zero_actual_offsets_use_legacy_name(self):
        self.assertEqual(
            synchronization_cache_suffix(
                np.zeros((8, 1)),
                np.zeros((8, 1)),
                drop_id=4,
                phase_std_deg=90,
                timing_std_samples=0.5,
            ),
            "",
        )

    def test_nonzero_offsets_include_drop_and_both_sweep_settings(self):
        self.assertEqual(
            synchronization_cache_suffix(
                np.array([[10.0]]),
                np.array([[0.0]]),
                drop_id=7,
                phase_std_deg=3.6,
                timing_std_samples=0,
            ),
            "_sync_drop_7_phase_std_deg_3p6_timing_std_samples_0",
        )
        self.assertEqual(
            synchronization_cache_suffix(
                np.array([[0.0]]),
                np.array([[2.0]]),
                drop_id=8,
                phase_std_deg=0,
                timing_std_samples=0.2,
            ),
            "_sync_drop_8_phase_std_deg_0_timing_std_samples_0p2",
        )

    def test_nonzero_offsets_without_metadata_bypass_disk_cache(self):
        self.assertIsNone(
            synchronization_cache_suffix(
                np.array([[1.0]]),
                np.array([[0.0]]),
            )
        )


class SynchronizationChannelCacheTest(unittest.TestCase):
    def setUp(self):
        self.predictor = standard_rc_pred_freq_mimo.__new__(
            standard_rc_pred_freq_mimo
        )
        self.predictor.ns3_config = SimpleNamespace(
            num_bs_ant=4,
            num_ue_ant=2,
            num_rxue_sel=4,
            num_txue_sel=8,
        )

    def _load(self, cache_base, cfo_vals, sto_vals, estimate, **metadata):
        with patch.object(
            rc_pred_freq_mimo,
            "lmmse_channel_estimation",
            side_effect=estimate,
        ):
            return self.predictor._load_or_estimate_channel(
                67,
                object(),
                object(),
                cfo_vals,
                sto_vals,
                str(cache_base),
                None,
                None,
                **metadata,
            )

    def test_zero_loads_legacy_and_nonzero_uses_isolated_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_base = Path(tmp) / "channel_estimates_drop_1"
            cache_dir = Path(f"{cache_base}_rx_12_tx_20")
            cache_dir.mkdir()
            np.savez(
                cache_dir / "dmimochans_67.npz",
                h_freq_csi=np.array([1.0]),
                err_var_csi=np.array([0.1]),
            )

            estimate_calls = []

            def estimate(*args, **kwargs):
                estimate_calls.append(kwargs)
                return np.array([2.0]), np.array([0.2])

            zero_h, _ = self._load(
                cache_base,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                estimate,
                sync_drop_id=1,
                sync_phase_std_deg=0,
                sync_timing_std_samples=0,
            )
            np.testing.assert_array_equal(zero_h, [[1.0]])
            self.assertEqual(estimate_calls, [])

            nonzero_metadata = dict(
                sync_drop_id=1,
                sync_phase_std_deg=3.6,
                sync_timing_std_samples=0,
            )
            nonzero_h, _ = self._load(
                cache_base,
                np.array([[10.0]]),
                np.zeros((1, 1)),
                estimate,
                **nonzero_metadata,
            )
            np.testing.assert_array_equal(nonzero_h, [[2.0]])
            self.assertEqual(len(estimate_calls), 1)
            sync_cache = cache_dir / (
                "dmimochans_67_sync_drop_1_phase_std_deg_3p6_"
                "timing_std_samples_0.npz"
            )
            self.assertTrue(sync_cache.is_file())

            self._load(
                cache_base,
                np.array([[10.0]]),
                np.zeros((1, 1)),
                estimate,
                **nonzero_metadata,
            )
            self.assertEqual(len(estimate_calls), 1)

    def test_nonzero_without_metadata_is_never_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_base = Path(tmp) / "channel_estimates_drop_1"
            estimate_calls = []

            def estimate(*args, **kwargs):
                estimate_calls.append(kwargs)
                return np.array([3.0]), np.array([0.3])

            for _ in range(2):
                self._load(
                    cache_base,
                    np.array([[10.0]]),
                    np.zeros((1, 1)),
                    estimate,
                )
            self.assertEqual(len(estimate_calls), 2)
            self.assertFalse(Path(f"{cache_base}_rx_12_tx_20").exists())


if __name__ == "__main__":
    unittest.main()
