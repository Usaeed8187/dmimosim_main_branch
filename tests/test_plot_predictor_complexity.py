import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _ROOT / "results" / "plot_predictor_complexity.py"
_SPEC = importlib.util.spec_from_file_location("plot_predictor_complexity", _MODULE_PATH)
complexity = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = complexity
_SPEC.loader.exec_module(complexity)


def _record(seconds, python_peak=1024, rss_peak=4096):
    return {
        "elapsed_seconds": seconds,
        "python_peak_increment_bytes": python_peak,
        "process_peak_rss_bytes": rss_peak,
    }


def _artifact(drop, workers, inference, configuration):
    return complexity.Artifact(
        path=Path(f"drop_{drop}_workers_{workers}.npz"),
        mobility="high_mobility",
        drop=drop,
        rx_ues=4,
        tx_ues=2,
        method="wesn_lite",
        workers=workers,
        split_mode="time_split",
        raw={
            "phases": {
                "inference_system": [_record(value) for value in inference],
                "configuration_system": [_record(configuration)],
            },
            "persistent_predictor_bytes": 2**20,
            "wesn_residue_rank_summary": {
                "histogram": {"1": 4, "2": 2},
                "energy_threshold": 0.95,
            },
        },
    )


class PredictorComplexityPlotTest(unittest.TestCase):
    def test_discovers_instrumented_npz(self):
        raw = {
            "phases": {
                "inference_system": [_record(0.1)],
                "configuration_system": [_record(1.0)],
            },
            "persistent_predictor_bytes": 1024,
        }
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folder = base / "channels_high_mobility_7"
            folder.mkdir()
            path = folder / (
                "mu_mimo_results_link_adapt_rx_UE_4_tx_UE_2_"
                "prediction_wesn_lite_pmi_quantization_True_time_split.npz"
            )
            np.savez(
                path,
                predictor_complexity_raw_json=np.asarray(json.dumps(raw)),
                predictor_workers=np.asarray(1),
            )
            artifacts, skipped = complexity.discover_artifacts(
                base,
                {"wesn_lite"},
                {"high_mobility"},
                4,
                "time_split",
                False,
            )

        self.assertEqual(skipped, 0)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].num_rus, 3)

    def test_pooling_amortization_and_parallel_efficiency(self):
        artifacts = [
            _artifact(1, 1, [1.0, 3.0], 10.0),
            _artifact(2, 1, [2.0, 4.0], 10.0),
            _artifact(1, 2, [1.0, 1.0], 10.0),
        ]
        summary = complexity.summarize_groups(complexity.group_artifacts(artifacts), None)
        q1 = next(row for row in summary if row["workers"] == 1)

        self.assertEqual(q1["num_rus"], 3)
        self.assertEqual(q1["num_latency_samples"], 4)
        self.assertAlmostEqual(q1["latency_p50_ms"], 2500.0)
        self.assertAlmostEqual(q1["amortized_latency_ms"], 7500.0)
        self.assertAlmostEqual(q1["persistent_memory_mean_mib"], 1.0)
        self.assertAlmostEqual(q1["residue_rank_mean"], 4.0 / 3.0)
        self.assertEqual(q1["residue_rank_mode"], 1)
        self.assertEqual(q1["residue_rank_count"], 12)
        self.assertAlmostEqual(q1["residue_energy_threshold"], 0.95)

        parallel = complexity.parallel_rows(summary)
        q2 = next(row for row in parallel if row["workers"] == 2)
        self.assertAlmostEqual(q2["speedup"], 2.5)
        self.assertAlmostEqual(q2["efficiency"], 1.25)

    def test_explicit_amortization_horizon(self):
        artifacts = [_artifact(1, 1, [1.0, 3.0], 10.0)]
        summary = complexity.summarize_groups(complexity.group_artifacts(artifacts), 5)
        self.assertAlmostEqual(summary[0]["amortized_latency_ms"], 4000.0)
        self.assertEqual(summary[0]["amortization_horizon"], "5")


if __name__ == "__main__":
    unittest.main()
