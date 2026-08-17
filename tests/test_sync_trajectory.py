import unittest
import importlib.util
from pathlib import Path
import sys

import numpy as np

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "dmimo" / "utils" / "sync_trajectory.py"
)
_SPEC = importlib.util.spec_from_file_location("sync_trajectory", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
generate_synchronization_trajectory = _MODULE.generate_synchronization_trajectory
SYNC_MODEL_VERSION = _MODULE.SYNC_MODEL_VERSION


class SynchronizationTrajectoryTest(unittest.TestCase):
    def _generate(self, **overrides):
        kwargs = dict(
            drop_id=7,
            num_mobile_rus=3,
            num_slots=100,
            frequency_std_ppb=3.73,
            initial_timing_std_ps=70.0,
            initial_phase_std_deg=2.0,
            phase_noise_s100_dbchz=None,
            carrier_frequency_hz=3.5e9,
            sample_rate_hz=7.68e6,
            slot_duration_s=1e-3,
            cyclic_prefix_samples=64,
            enabled=True,
        )
        kwargs.update(overrides)
        return generate_synchronization_trajectory(**kwargs)

    def test_same_drop_reproduces_complete_trajectory(self):
        first = self._generate(phase_noise_s100_dbchz=-90.0)
        second = self._generate(phase_noise_s100_dbchz=-90.0)
        np.testing.assert_array_equal(first.phase_offset_rad, second.phase_offset_rad)
        np.testing.assert_array_equal(
            first.timing_offset_samples, second.timing_offset_samples
        )

    def test_fixed_fractional_error_gives_linear_timing_and_phase(self):
        trajectory = self._generate()
        timing_steps = np.diff(trajectory.timing_offset_samples, axis=0)
        expected_timing_step = (
            trajectory.sample_rate_hz
            * trajectory.slot_duration_s
            * trajectory.fractional_frequency_error
        )
        np.testing.assert_allclose(
            timing_steps,
            np.broadcast_to(expected_timing_step, timing_steps.shape),
        )

        phase_steps = np.diff(trajectory.phase_offset_rad, axis=0)
        expected_phase_step = (
            2.0
            * np.pi
            * trajectory.carrier_frequency_hz
            * trajectory.slot_duration_s
            * trajectory.fractional_frequency_error
        )
        np.testing.assert_allclose(
            phase_steps,
            np.broadcast_to(expected_phase_step, phase_steps.shape),
        )

    def test_disabled_trajectory_is_zero_and_uses_versioned_cache_suffix(self):
        trajectory = self._generate(enabled=False, phase_noise_s100_dbchz=-80.0)
        self.assertFalse(trajectory.enabled)
        self.assertIn("_sync_clock_v2_drop_7_", trajectory.cache_suffix())
        self.assertIn("_pn_s100_dbchz_off", trajectory.cache_suffix())
        np.testing.assert_array_equal(trajectory.phase_offset_rad, 0.0)
        np.testing.assert_array_equal(trajectory.timing_offset_samples, 0.0)

    def test_merlo_scale_stays_well_inside_cp(self):
        trajectory = self._generate()
        self.assertLess(
            np.max(np.abs(trajectory.timing_offset_samples)),
            trajectory.cyclic_prefix_samples,
        )

    def test_s100_generates_nonzero_wiener_phase_innovations(self):
        deterministic = self._generate(phase_noise_s100_dbchz=None)
        noisy = self._generate(phase_noise_s100_dbchz=-80.0)
        self.assertGreater(noisy.phase_noise_std_deg_per_slot, 0.0)
        self.assertFalse(
            np.array_equal(noisy.phase_offset_rad, deterministic.phase_offset_rad)
        )

    def test_s100_conversion_matches_ngo_larsson_equation_23(self):
        trajectory = self._generate(
            frequency_std_ppb=0.0,
            initial_timing_std_ps=0.0,
            initial_phase_std_deg=0.0,
            phase_noise_s100_dbchz=-120.0,
        )
        expected_deg_per_slot = 360.0 * np.sqrt(
            (100e3) ** 2 * 10.0 ** (-120.0 / 10.0) * 1e-3
        )
        self.assertAlmostEqual(
            trajectory.phase_noise_std_deg_per_slot,
            expected_deg_per_slot,
            places=12,
        )
        self.assertEqual(trajectory.model_version, "clock_v2")
        self.assertEqual(SYNC_MODEL_VERSION, "clock_v2")


if __name__ == "__main__":
    unittest.main()
