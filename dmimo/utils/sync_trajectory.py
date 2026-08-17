"""Residual synchronization trajectories for distributed OFDM transmitters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


SYNC_MODEL_VERSION = "clock_v2"

# Ngo--Larsson Eq. (23) specifies oscillator phase noise at a 100 kHz
# frequency offset. Squaring 100 kHz gives the 1e10 factor in the conversion
# from S_100 [dBc/Hz] to discrete-time Wiener increment variance.
PHASE_NOISE_REFERENCE_OFFSET_HZ = 100e3


def _token(value: float) -> str:
    return format(float(value), "g").replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class SynchronizationTrajectory:
    """One reproducible post-synchronization clock trajectory per drop.

    Arrays contain only the non-reference RUs. RU 0 is inserted as an ideal
    reference when the offsets are applied to an OFDM resource grid.
    """

    drop_id: int
    fractional_frequency_error: np.ndarray
    initial_timing_offset_ps: np.ndarray
    initial_timing_offset_samples: np.ndarray
    initial_phase_offset_rad: np.ndarray
    timing_offset_samples: np.ndarray
    phase_offset_rad: np.ndarray
    frequency_std_ppb: float
    initial_timing_std_ps: float
    initial_phase_std_deg: float
    phase_noise_s100_dbchz: Optional[float]
    phase_noise_std_deg_per_slot: float
    carrier_frequency_hz: float
    sample_rate_hz: float
    slot_duration_s: float
    cyclic_prefix_samples: int
    model_version: str = SYNC_MODEL_VERSION

    @property
    def num_slots(self) -> int:
        return int(self.phase_offset_rad.shape[0])

    @property
    def enabled(self) -> bool:
        return bool(
            self.frequency_std_ppb != 0
            or self.initial_timing_std_ps != 0
            or self.initial_phase_std_deg != 0
            or self.phase_noise_s100_dbchz is not None
        )

    def state_at(self, slot_idx: int) -> tuple[np.ndarray, np.ndarray]:
        slot_idx = int(slot_idx)
        if slot_idx < 0 or slot_idx >= self.num_slots:
            raise IndexError(
                f"Synchronization slot {slot_idx} is outside [0, {self.num_slots})."
            )
        return self.phase_offset_rad[slot_idx], self.timing_offset_samples[slot_idx]

    def cache_suffix(self) -> str:
        pn_token = (
            "off"
            if self.phase_noise_s100_dbchz is None
            else _token(self.phase_noise_s100_dbchz)
        )
        return (
            f"_sync_{self.model_version}_drop_{self.drop_id}"
            f"_freq_std_ppb_{_token(self.frequency_std_ppb)}"
            f"_timing0_std_ps_{_token(self.initial_timing_std_ps)}"
            f"_phase0_std_deg_{_token(self.initial_phase_std_deg)}"
            f"_pn_s100_dbchz_{pn_token}"
        )

    def metadata(self) -> dict[str, np.ndarray]:
        return {
            "sync_model_version": np.asarray(self.model_version),
            "sync_drop_id": np.asarray(self.drop_id),
            "sync_frequency_std_ppb": np.asarray(self.frequency_std_ppb),
            "sync_initial_timing_std_ps": np.asarray(self.initial_timing_std_ps),
            "sync_initial_phase_std_deg": np.asarray(self.initial_phase_std_deg),
            "sync_phase_noise_s100_dbchz": np.asarray(
                np.nan
                if self.phase_noise_s100_dbchz is None
                else self.phase_noise_s100_dbchz
            ),
            "sync_phase_noise_std_deg_per_slot": np.asarray(
                self.phase_noise_std_deg_per_slot
            ),
            "sync_phase_noise_reference_offset_hz": np.asarray(
                PHASE_NOISE_REFERENCE_OFFSET_HZ
            ),
            "sync_carrier_frequency_hz": np.asarray(self.carrier_frequency_hz),
            "sync_sample_rate_hz": np.asarray(self.sample_rate_hz),
            "sync_slot_duration_s": np.asarray(self.slot_duration_s),
            "residual_fractional_frequency_error": np.asarray(
                self.fractional_frequency_error
            ),
            "residual_initial_timing_offset_ps": np.asarray(
                self.initial_timing_offset_ps
            ),
            "residual_initial_timing_offset_samples": np.asarray(
                self.initial_timing_offset_samples
            ),
            "residual_initial_phase_offset_rad": np.asarray(
                self.initial_phase_offset_rad
            ),
            "residual_timing_offset_samples_per_slot": np.asarray(
                self.timing_offset_samples
            ),
            "residual_phase_offset_rad_per_slot": np.asarray(
                self.phase_offset_rad
            ),
        }


def generate_synchronization_trajectory(
    *,
    drop_id: int,
    num_mobile_rus: int,
    num_slots: int,
    frequency_std_ppb: float,
    initial_timing_std_ps: float,
    initial_phase_std_deg: float,
    phase_noise_s100_dbchz: Optional[float],
    carrier_frequency_hz: float,
    sample_rate_hz: float,
    slot_duration_s: float,
    cyclic_prefix_samples: int,
    enabled: bool = True,
) -> SynchronizationTrajectory:
    """Generate the Merlo-inspired clock model with Ngo-Larsson phase noise."""

    if min(frequency_std_ppb, initial_timing_std_ps, initial_phase_std_deg) < 0:
        raise ValueError("Synchronization standard deviations must be nonnegative.")
    if num_mobile_rus < 0 or num_slots <= 0:
        raise ValueError("num_mobile_rus must be nonnegative and num_slots positive.")

    if not enabled:
        frequency_std_ppb = 0.0
        initial_timing_std_ps = 0.0
        initial_phase_std_deg = 0.0
        phase_noise_s100_dbchz = None

    rng = np.random.default_rng(int(drop_id))
    shape = (int(num_mobile_rus), 1)
    fractional_error = rng.normal(
        0.0, float(frequency_std_ppb) * 1e-9, size=shape
    )
    timing0_ps = rng.normal(0.0, float(initial_timing_std_ps), size=shape)
    timing0_samples = timing0_ps * 1e-12 * float(sample_rate_hz)
    phase0_rad = rng.normal(
        0.0, np.deg2rad(float(initial_phase_std_deg)), size=shape
    )

    slot_numbers = np.arange(int(num_slots), dtype=np.float64)[:, None, None]
    samples_per_slot = float(sample_rate_hz) * float(slot_duration_s)
    timing_samples = timing0_samples[None, ...] + (
        slot_numbers * samples_per_slot * fractional_error[None, ...]
    )

    deterministic_phase = phase0_rad[None, ...] + (
        2.0
        * np.pi
        * float(carrier_frequency_hz)
        * float(slot_duration_s)
        * slot_numbers
        * fractional_error[None, ...]
    )

    phase_noise_std_rad_per_slot = 0.0
    accumulated_phase_noise = np.zeros_like(deterministic_phase)
    if phase_noise_s100_dbchz is not None and num_slots > 1:
        # Ngo-Larsson Eq. (23) gives the Wiener increment variance per
        # baseband sample. Independent increments add over one slot.
        variance_per_sample = (
            4.0
            * np.pi**2
            * PHASE_NOISE_REFERENCE_OFFSET_HZ**2
            * 10.0 ** (float(phase_noise_s100_dbchz) / 10.0)
            / float(sample_rate_hz)
        )
        phase_noise_std_rad_per_slot = np.sqrt(
            variance_per_sample * samples_per_slot
        )
        increments = rng.normal(
            0.0,
            phase_noise_std_rad_per_slot,
            size=(int(num_slots) - 1, *shape),
        )
        accumulated_phase_noise[1:] = np.cumsum(increments, axis=0)

    phase_rad = deterministic_phase + accumulated_phase_noise
    if np.any(np.abs(timing_samples) >= int(cyclic_prefix_samples)):
        maximum = float(np.max(np.abs(timing_samples)))
        raise ValueError(
            "Residual timing trajectory must remain strictly within the cyclic "
            f"prefix ({maximum:g} >= {cyclic_prefix_samples} samples)."
        )

    return SynchronizationTrajectory(
        drop_id=int(drop_id),
        fractional_frequency_error=fractional_error,
        initial_timing_offset_ps=timing0_ps,
        initial_timing_offset_samples=timing0_samples,
        initial_phase_offset_rad=phase0_rad,
        timing_offset_samples=timing_samples,
        phase_offset_rad=phase_rad,
        frequency_std_ppb=float(frequency_std_ppb),
        initial_timing_std_ps=float(initial_timing_std_ps),
        initial_phase_std_deg=float(initial_phase_std_deg),
        phase_noise_s100_dbchz=(
            None
            if phase_noise_s100_dbchz is None
            else float(phase_noise_s100_dbchz)
        ),
        phase_noise_std_deg_per_slot=float(
            np.rad2deg(phase_noise_std_rad_per_slot)
        ),
        carrier_frequency_hz=float(carrier_frequency_hz),
        sample_rate_hz=float(sample_rate_hz),
        slot_duration_s=float(slot_duration_s),
        cyclic_prefix_samples=int(cyclic_prefix_samples),
    )
