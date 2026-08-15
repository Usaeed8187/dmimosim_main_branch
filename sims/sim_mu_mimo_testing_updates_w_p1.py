"""Phase-1-enabled entry point for the current MU-MIMO throughput pipeline.

All phase-2 channel prediction, offline configuration, reporting, and output
handling live in ``sim_mu_mimo_testing_updates`` so the phase-1 and phase-2-only
sweeps cannot drift apart.
"""

import os
import sys
from pathlib import Path


os.environ.setdefault("DMIMO_PHASE_1_ENABLED", "True")
os.environ.setdefault("DMIMO_PHASE_3_ENABLED", "False")
os.environ.setdefault("DMIMO_NUM_SLOTS_P1", "2")
os.environ.setdefault("DMIMO_NUM_SLOTS_P2", "2")

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from sims.sim_mu_mimo_testing_updates import log_error, run_simulation


if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as exc:  # noqa: BLE001
        log_error(exc)
        sys.exit(1)
