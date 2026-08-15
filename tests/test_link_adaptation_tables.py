import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np


# Load this focused module without importing dmimo/__init__.py.  The latter
# eagerly imports optional predictor dependencies that are unrelated to MCS
# table validation and are not installed in every test environment.
_fake_dmimo = types.ModuleType("dmimo")
_fake_dmimo.__path__ = []
_fake_mimo = types.ModuleType("dmimo.mimo")
_fake_mimo.rankAdaptation = object
_module_path = Path(__file__).parents[1] / "dmimo/mimo/link_adaptation.py"
_spec = importlib.util.spec_from_file_location("_link_adaptation_tables", _module_path)
_module = importlib.util.module_from_spec(_spec)
with mock.patch.dict(
    sys.modules,
    {"dmimo": _fake_dmimo, "dmimo.mimo": _fake_mimo},
):
    _spec.loader.exec_module(_module)

get_link_adaptation_table = _module.get_link_adaptation_table
project_mcs_indices_to_sionna_supported = (
    _module.project_mcs_indices_to_sionna_supported
)
sionna_ldpc5g_supported_mcs_mask = _module.sionna_ldpc5g_supported_mcs_mask


class TestLinkAdaptationTables(unittest.TestCase):
    def test_38_214_table_1_exact_entries(self):
        beta, thresholds, candidates = get_link_adaptation_table("38.214")

        self.assertEqual(candidates.shape, (29, 2))
        self.assertEqual(beta.shape, (29,))
        self.assertEqual(thresholds.shape, (29,))
        np.testing.assert_array_equal(candidates[:, 0].astype(int), [2] * 10 + [4] * 7 + [6] * 12)
        self.assertAlmostEqual(candidates[0, 1], 120 / 1024)
        self.assertAlmostEqual(candidates[9, 1], 679 / 1024)
        self.assertAlmostEqual(candidates[10, 1], 340 / 1024)
        self.assertAlmostEqual(candidates[16, 1], 658 / 1024)
        self.assertAlmostEqual(candidates[17, 1], 438 / 1024)
        self.assertAlmostEqual(candidates[28, 1], 948 / 1024)

    def test_legacy_tables_remain_available(self):
        for name, candidate_count in (("short", 3), ("long", 9)):
            beta, thresholds, candidates = get_link_adaptation_table(name)
            self.assertEqual(beta.shape, thresholds.shape)
            self.assertEqual(candidates.shape, (candidate_count, 2))

    def test_unknown_table_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown link-adaptation table"):
            get_link_adaptation_table("not-a-table")

    def test_highest_38_214_mcs_is_selectable(self):
        _, thresholds, candidates = get_link_adaptation_table("38.214")
        result = _module.linkAdaptation.lookup_table(
            object(),
            thresholds + 1.0,
            thresholds,
            candidates,
            return_mcs_index=True,
        )
        self.assertEqual(result[3], 28)
        self.assertEqual(result[0], 6)
        self.assertAlmostEqual(result[1], 948 / 1024)

    def test_current_phy_floors_selection_at_supported_mcs_3(self):
        _, thresholds, candidates = get_link_adaptation_table("38.214")
        predictor = types.SimpleNamespace(minimum_mcs_index=3)
        result = _module.linkAdaptation.lookup_table(
            predictor,
            thresholds - 100.0,
            thresholds,
            candidates,
            return_mcs_index=True,
        )
        self.assertEqual(result[3], 3)
        self.assertEqual(result[0], 2)
        self.assertAlmostEqual(result[1], 251 / 1024)

    def test_sionna_projection_skips_unsupported_noncontiguous_rates(self):
        _, _, candidates = get_link_adaptation_table("38.214")
        supported = sionna_ldpc5g_supported_mcs_mask(candidates, 11952)

        self.assertTrue(supported[9])
        self.assertFalse(supported[10])
        self.assertTrue(supported[11])
        self.assertTrue(supported[23])
        self.assertFalse(supported[24])
        np.testing.assert_array_equal(
            project_mcs_indices_to_sionna_supported(
                np.array([0, 10, 24, 28]), candidates, 11952
            ),
            np.array([3, 9, 23, 23]),
        )


if __name__ == "__main__":
    unittest.main()
