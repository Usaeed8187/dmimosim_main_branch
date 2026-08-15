"""Compatibility import for the phase-1-enabled MU-MIMO pipeline.

Phase 1 is selected with ``cfg.phase_1_enabled`` in the shared implementation.
Keeping one implementation ensures every phase-2 predictor follows the same
configuration and evaluation path in both throughput pipelines.
"""

from .mu_mimo_testing_updates_v2 import MU_MIMO, sim_mu_mimo, sim_mu_mimo_all

__all__ = ["MU_MIMO", "sim_mu_mimo", "sim_mu_mimo_all"]
