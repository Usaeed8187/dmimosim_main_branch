"""Compatibility shim after renaming the SNR sweep script."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ns3_channels_prediction_across_snr import *  # noqa: F401,F403
