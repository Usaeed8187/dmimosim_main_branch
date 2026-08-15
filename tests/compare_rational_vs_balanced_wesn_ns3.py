"""Compare rational-fit and balanced-truncation WESN configuration on ns-3 CSI.

Both methods use exactly the same offline Kalman-predictor ensemble and PCA
modes.  The branches differ only after PCA:

* rational: the existing shared-denominator vector rational fit;
* balanced: exact rational lifting of each empirical PCA mode followed by
  order-K square-root balanced truncation.

The reservoirs are linear and use the same M modes, K poles per mode, current
input skip features, online ridge readout, channel samples, and noise draws.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag, solve_discrete_lyapunov

from test_ns3_channels_prediction import (
    ConfiguredWeightsESN,
    add_complex_awgn,
    build_augmented_obs_matrix,
    channels_to_tiles,
    collect_esn_states_per_tile,
    fit_shared_denominator_vector_rational,
    decompose_rp_fit_into_first_order,
    load_clean_p2p_channels,
    solve_riccati_steady_state_complex,
    steady_state_predictor_transfer_samples_from_kalman,
    vectorize_transfer_samples,
)
from dmimo.channel.kalman_filter_pred import kalman_filter_pred


@dataclass
class KalmanTransferRealization:
    """One steady-state predictor in conventional state-space form."""

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray

    def frequency_response(self, omegas: np.ndarray) -> np.ndarray:
        eye = np.eye(self.a.shape[0], dtype=np.complex128)
        out = np.empty(
            (omegas.size, self.c.shape[0], self.b.shape[1]),
            dtype=np.complex128,
        )
        for idx, omega in enumerate(omegas):
            z = np.exp(1j * omega)
            out[idx] = self.d + self.c @ np.linalg.solve(
                z * eye - self.a, self.b
            )
        return out


class RankFactoredIIRReservoir:
    """Linear scalar-state IIR bank obtained from rank-one MIMO residues."""

    def __init__(
        self,
        poles: np.ndarray,
        residues: np.ndarray,
        d: int,
        rank_tolerance: float = 1e-6,
    ) -> None:
        recurrent_weights = []
        input_rows = []
        for mode_idx in range(poles.shape[0]):
            for pole_idx in range(poles.shape[1]):
                residue = residues[mode_idx, :, pole_idx].reshape(d, d).T
                _, singular_values, vh = np.linalg.svd(
                    residue, full_matrices=False
                )
                leading = float(singular_values[0])
                tail = float(np.linalg.norm(singular_values[1:]))
                if tail > rank_tolerance * max(leading, 1e-15):
                    raise RuntimeError(
                        "A simple pole of the balanced realization has a "
                        f"numerically non-rank-one residue (relative tail "
                        f"{tail / max(leading, 1e-15):.3e})."
                    )
                recurrent_weights.append(poles[mode_idx, pole_idx])
                input_rows.append(singular_values[0] * vh[0])

        self.w_res = np.asarray(recurrent_weights, dtype=np.complex128)
        self.w_in = np.asarray(input_rows, dtype=np.complex128)
        self.state_dim = int(self.w_res.size)

    def collect_states(self, history: np.ndarray) -> np.ndarray:
        t_len, num_tiles, _ = history.shape
        states = np.zeros(
            (t_len, num_tiles, self.state_dim), dtype=np.complex128
        )
        state = np.zeros((num_tiles, self.state_dim), dtype=np.complex128)
        for time_idx in range(t_len):
            driven = history[time_idx].astype(np.complex128) @ self.w_in.T
            state = state * self.w_res[None, :] + driven
            states[time_idx] = state
        return states


def _node_antenna_slice(node_idx: int, num_bs_ant: int = 4, num_ue_ant: int = 2) -> slice:
    if node_idx == 0:
        return slice(0, num_bs_ant)
    start = num_bs_ant + (node_idx - 1) * num_ue_ant
    return slice(start, start + num_ue_ant)


def load_production_channel_sequence(
    ns3_root: Path,
    mobility: str,
    drop_idx: int,
    slots: np.ndarray,
    num_rx_ues: int,
    num_tx_ues: int,
    num_bs_ant: int = 4,
    num_ue_ant: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the same normalized truth and saved LMMSE CSI used by MU-MIMO."""
    total_rx_ant = num_bs_ant + num_ue_ant * num_rx_ues
    total_tx_ant = num_bs_ant + num_ue_ant * num_tx_ues
    raw_folder = ns3_root / f"channels_{mobility}_{drop_idx}"
    estimate_folder = ns3_root / (
        f"channel_estimates_{mobility}_drop_{drop_idx}_"
        f"rx_{total_rx_ant}_tx_{total_tx_ant}"
    )
    if not raw_folder.exists():
        raise FileNotFoundError(f"Raw ns-3 channel folder not found: {raw_folder}")
    if not estimate_folder.exists():
        raise FileNotFoundError(
            "Saved production channel-estimate folder not found: "
            f"{estimate_folder}"
        )

    clean_sequence = []
    observed_sequence = []
    error_variance_sequence = []
    for slot in slots:
        raw_path = raw_folder / f"dmimochans_{int(slot)}.npz"
        estimate_path = estimate_folder / f"dmimochans_{int(slot)}.npz"
        if not raw_path.exists() or not estimate_path.exists():
            raise FileNotFoundError(
                f"Missing raw/estimated channel pair for slot {slot}: "
                f"{raw_path}, {estimate_path}"
            )

        with np.load(raw_path) as data:
            raw_channel = np.asarray(
                data["Hdm"][:total_rx_ant, :total_tx_ant],
                dtype=np.complex128,
            )
            path_loss = np.asarray(
                data["Ldm"][: num_rx_ues + 1, : num_tx_ues + 1],
                dtype=np.float64,
            )

        # Match LoadNs3Channel.convert_channel("dMIMO"). The default ns-3
        # configuration uses 35/26 dBm transmit powers and 5 dBi gains.
        tx_power = np.concatenate(
            [np.asarray([40.0]), np.full(num_tx_ues, 31.0)]
        )
        rx_gain = np.full(num_rx_ues + 1, 5.0)
        received_path_power = (
            tx_power[None, :, None]
            + rx_gain[:, None, None]
            - path_loss
        )
        total_received_power = np.log10(
            np.sum(10.0 ** received_path_power, axis=1, keepdims=True)
        )
        agc_db = received_path_power - total_received_power
        agc_db = np.concatenate(
            [
                np.repeat(agc_db[:1], num_bs_ant, axis=0),
                np.repeat(agc_db[1:], num_ue_ant, axis=0),
            ],
            axis=0,
        )
        agc_db = np.concatenate(
            [
                np.repeat(agc_db[:, :1], num_bs_ant, axis=1),
                np.repeat(agc_db[:, 1:], num_ue_ant, axis=1),
            ],
            axis=1,
        )
        clean_sequence.append(
            raw_channel * np.sqrt(10.0 ** (agc_db / 10.0))[..., None]
        )

        with np.load(estimate_path) as data:
            estimate = np.asarray(data["h_freq_csi"], dtype=np.complex128)
            error_variance = np.asarray(
                data["err_var_csi"], dtype=np.float64
            )
        observed_sequence.append(estimate[0, 0, :, 0])
        error_variance_sequence.append(error_variance[0, 0, :, 0])

    clean = np.stack(clean_sequence)
    observed = np.stack(observed_sequence)
    error_variance = np.stack(error_variance_sequence)
    if clean.shape != observed.shape:
        raise RuntimeError(
            f"Normalized truth shape {clean.shape} differs from estimate shape "
            f"{observed.shape}."
        )
    return clean, observed, error_variance


def collect_transfer_ensemble(
    tiles_noisy: np.ndarray,
    ar_order: int,
    history_len: int,
    r_diag: np.ndarray,
    num_freqs: int,
) -> tuple[list[KalmanTransferRealization], np.ndarray, np.ndarray]:
    """Build all offline steady-state Kalman systems and sampled responses."""
    t_dec, _, d = tiles_noisy.shape
    p_eff = min(ar_order, history_len - 1)
    helper = kalman_filter_pred(ar_order=ar_order)
    omegas = np.linspace(0.0, np.pi, num_freqs, endpoint=True)
    realizations: list[KalmanTransferRealization] = []
    sampled_vectors = []

    for start in range(t_dec - history_len):
        history = tiles_noisy[start : start + history_len]
        a_blocks, q_proc = helper._estimate_ar_p_q_joint(history, p_eff)
        a_blocks = [a.conj() for a in a_blocks]
        f_aug, q_aug = helper._build_augmented_system(a_blocks, q_proc)
        c_obs = build_augmented_obs_matrix(d, p_eff)
        r_mat = np.diag(np.maximum(r_diag, 1e-12).astype(np.complex128))
        _, k_ss = solve_riccati_steady_state_complex(
            f_aug, q_aug, c_obs, r_mat
        )

        # Predictor used by the existing implementation:
        # T(z)=C_obs (I-A_p z^-1)^-1 B_p.
        a_p = f_aug - f_aug @ k_ss @ c_obs
        b_p = f_aug @ k_ss

        # Equivalent conventional realization:
        # T(z)=D_std+C_std (zI-A_p)^-1 B_p.
        c_std = c_obs @ a_p
        d_std = c_obs @ b_p
        realization = KalmanTransferRealization(a_p, b_p, c_std, d_std)
        realizations.append(realization)

        h_samples = steady_state_predictor_transfer_samples_from_kalman(
            f_aug, k_ss, d=d, num_freqs=num_freqs
        )
        sampled_vectors.append(vectorize_transfer_samples(h_samples))

        # Catch indexing/convention errors before performing PCA.
        h_standard = realization.frequency_response(omegas)
        relative_error = np.linalg.norm(h_samples - h_standard) / max(
            np.linalg.norm(h_samples), 1e-15
        )
        if relative_error > 1e-7:
            raise RuntimeError(
                "Conventional and z^-1 Kalman transfer forms disagree: "
                f"relative error={relative_error:.3e}."
            )

    if not realizations:
        raise ValueError("No offline Kalman transfer realizations were produced.")
    return realizations, np.stack(sampled_vectors, axis=1), omegas


def pca_modes_and_lift_coefficients(
    sampled_vectors: np.ndarray,
    num_modes: int,
    rank_tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, PCA modes, eigenvalues, and exact lift coefficients."""
    mean_v = np.mean(sampled_vectors, axis=1, keepdims=True)
    vc = sampled_vectors - mean_v
    u, singular_values, vh = np.linalg.svd(vc, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        raise ValueError("The centered transfer ensemble has zero numerical rank.")
    numerical_rank = int(
        np.count_nonzero(singular_values > rank_tol * singular_values[0])
    )
    if num_modes > numerical_rank:
        raise ValueError(
            f"Requested M={num_modes}, but centered transfer ensemble has "
            f"numerical rank {numerical_rank}."
        )

    q_modes = u[:, :num_modes]
    svals = singular_values[:num_modes]
    eigenvalues = svals**2 / sampled_vectors.shape[1]
    right_vectors = vh.conj().T[:, :num_modes]
    alpha = right_vectors / svals[None, :]

    reconstruction = vc @ alpha
    relative_error = np.linalg.norm(reconstruction - q_modes) / max(
        np.linalg.norm(q_modes), 1e-15
    )
    if relative_error > 1e-8:
        raise RuntimeError(
            f"PCA lift coefficient check failed: relative error={relative_error:.3e}."
        )
    return mean_v, q_modes, eigenvalues, alpha


def _psd_factor(matrix: np.ndarray, relative_tol: float) -> np.ndarray:
    """Return R with matrix approximately R R^H, dropping numerical zeros."""
    hermitian = 0.5 * (matrix + matrix.conj().T)
    values, vectors = np.linalg.eigh(hermitian)
    largest = max(float(np.max(values)), 0.0)
    if largest == 0.0:
        return np.zeros((matrix.shape[0], 0), dtype=np.complex128)
    keep = values > relative_tol * largest
    return vectors[:, keep] * np.sqrt(values[keep])[None, :]


def square_root_balanced_truncate(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    order: int,
    rank_tol: float = 1e-12,
) -> tuple[KalmanTransferRealization, np.ndarray, float]:
    """Stable discrete-time square-root balanced truncation."""
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(a))))
    if spectral_radius >= 1.0 - 1e-10:
        raise ValueError(
            "Balanced truncation requires a stable lifted system; "
            f"spectral radius is {spectral_radius:.9f}."
        )

    controllability = solve_discrete_lyapunov(a, b @ b.conj().T)
    observability = solve_discrete_lyapunov(
        a.conj().T, c.conj().T @ c
    )
    rc = _psd_factor(controllability, rank_tol)
    ro = _psd_factor(observability, rank_tol)
    if rc.shape[1] == 0 or ro.shape[1] == 0:
        raise ValueError("Lifted PCA mode has no controllable-observable dynamics.")

    u, hsv, vh = np.linalg.svd(ro.conj().T @ rc, full_matrices=False)
    numerical_rank = int(np.count_nonzero(hsv > rank_tol * hsv[0]))
    if order > numerical_rank:
        raise ValueError(
            f"Requested K={order}, but lifted mode has only {numerical_rank} "
            "nonzero Hankel singular values."
        )

    u_k = u[:, :order]
    v_k = vh.conj().T[:, :order]
    inv_sqrt = 1.0 / np.sqrt(hsv[:order])
    right_projection = (rc @ v_k) * inv_sqrt[None, :]
    left_projection = (ro @ u_k) * inv_sqrt[None, :]

    a_red = left_projection.conj().T @ a @ right_projection
    b_red = left_projection.conj().T @ b
    c_red = c @ right_projection
    reduced = KalmanTransferRealization(a_red, b_red, c_red, d.copy())
    error_bound = 2.0 * float(np.sum(hsv[order:numerical_rank]))
    return reduced, hsv[:numerical_rank], error_bound


def exact_lift_realization(
    realizations: list[KalmanTransferRealization],
    alpha_m: np.ndarray,
) -> KalmanTransferRealization:
    """Construct the exact continuous rational lift of one centered PCA mode."""
    # sum alpha_s (T_s - mean(T)) = sum beta_s T_s.
    beta = alpha_m - np.mean(alpha_m)
    a_raw = block_diag(*(system.a for system in realizations)).astype(
        np.complex128
    )
    b_raw = np.vstack([system.b for system in realizations])
    c_raw = np.hstack(
        [weight * system.c for weight, system in zip(beta, realizations)]
    )
    d_raw = sum(
        (weight * system.d for weight, system in zip(beta, realizations)),
        start=np.zeros_like(realizations[0].d),
    )
    return KalmanTransferRealization(a_raw, b_raw, c_raw, d_raw)


def residues_at_fixed_poles(
    response_samples: np.ndarray,
    poles: np.ndarray,
    omegas: np.ndarray,
    include_direct: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Represent sampled D-by-D response using fixed simple-pole functions."""
    zinv = np.exp(-1j * omegas)
    pole_basis = 1.0 / (1.0 - zinv[:, None] * poles[None, :])
    design = (
        np.column_stack([np.ones(omegas.size), pole_basis])
        if include_direct
        else pole_basis
    )
    values = response_samples.transpose(0, 2, 1).reshape(omegas.size, -1)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    if include_direct:
        direct = coefficients[0]
        residues = coefficients[1:].T
    else:
        direct = np.zeros(values.shape[1], dtype=np.complex128)
        residues = coefficients.T
    return residues, direct


def state_space_to_simple_iir_terms(
    system: KalmanTransferRealization,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a diagonalizable conventional realization to simple IIR terms."""
    poles, right_vectors = np.linalg.eig(system.a)
    inverse_right = np.linalg.inv(right_vectors)
    if np.min(np.abs(poles)) < 1e-10:
        raise RuntimeError(
            "A balanced pole is numerically zero; an exact realization then "
            "requires an explicit delay state rather than the current simple-IIR form."
        )
    residue_matrices = []
    for pole_idx, pole in enumerate(poles):
        conventional_residue = np.outer(
            system.c @ right_vectors[:, pole_idx],
            inverse_right[pole_idx] @ system.b,
        )
        # G/(z-p)=(G/p)[1/(1-p z^-1)-1].
        residue_matrices.append(conventional_residue / pole)
    residue_matrices = np.stack(residue_matrices, axis=0)
    direct = system.d - np.sum(residue_matrices, axis=0)
    residues = np.stack(
        [matrix.T.reshape(-1) for matrix in residue_matrices], axis=1
    )
    direct_vector = direct.T.reshape(-1)
    return poles, residues, direct_vector


def evaluate_iir_response(
    poles: np.ndarray,
    residues: np.ndarray,
    direct: np.ndarray,
    omegas: np.ndarray,
    d: int,
) -> np.ndarray:
    """Evaluate D-by-D partial-fraction response using repository vec order."""
    zinv = np.exp(-1j * omegas)
    basis = 1.0 / (1.0 - zinv[:, None] * poles[None, :])
    values = direct[None, :] + basis @ residues.T
    return values.reshape(omegas.size, d, d).transpose(0, 2, 1)


def transfer_approximation_metrics(
    reference: np.ndarray,
    approximation: np.ndarray,
) -> dict[str, float]:
    """Return absolute/relative worst-case and normalized L2 grid errors."""
    if reference.shape != approximation.shape:
        raise ValueError("Reference and approximation response shapes differ.")
    differences = reference - approximation
    error_singular_values = np.asarray(
        [
            np.linalg.svd(difference, compute_uv=False)[0]
            for difference in differences
        ]
    )
    reference_singular_values = np.asarray(
        [np.linalg.svd(value, compute_uv=False)[0] for value in reference]
    )
    hinf_error = float(np.max(error_singular_values))
    reference_hinf = float(np.max(reference_singular_values))
    l2_error = float(
        np.sqrt(np.mean(np.linalg.norm(differences, axis=(1, 2)) ** 2))
    )
    reference_l2 = float(
        np.sqrt(np.mean(np.linalg.norm(reference, axis=(1, 2)) ** 2))
    )
    return {
        "hinf_absolute": hinf_error,
        "hinf_relative": hinf_error / max(reference_hinf, 1e-15),
        "l2_absolute": l2_error,
        "l2_relative": l2_error / max(reference_l2, 1e-15),
    }


def build_shared_configurations(
    realizations: list[KalmanTransferRealization],
    sampled_vectors: np.ndarray,
    omegas: np.ndarray,
    num_modes: int,
    order: int,
    d: int,
    comparison_num_freqs: int = 2048,
) -> tuple[ConfiguredWeightsESN, RankFactoredIIRReservoir, dict]:
    """Build current and balanced reservoirs from one shared PCA."""
    _, q_modes, pca_eigenvalues, alpha = pca_modes_and_lift_coefficients(
        sampled_vectors, num_modes
    )
    value_dim = d * d
    num_freqs = omegas.size
    # The current ConfiguredWeightsESN creates D scalar states for every
    # matrix-valued pole residue. Match that actual recurrent-state budget.
    balanced_order = order * d

    rational_poles = np.empty((num_modes, order), dtype=np.complex128)
    rational_residues = np.empty(
        (num_modes, value_dim, order), dtype=np.complex128
    )
    balanced_poles = np.empty(
        (num_modes, balanced_order), dtype=np.complex128
    )
    balanced_residues = np.empty(
        (num_modes, value_dim, balanced_order), dtype=np.complex128
    )
    balanced_direct = np.empty(
        (num_modes, value_dim), dtype=np.complex128
    )
    bounds = []
    sampled_errors = []
    partial_fraction_errors = []
    rational_pca_grid_metrics = []
    balanced_pca_grid_metrics = []
    rational_dense_grid_metrics = []
    balanced_dense_grid_metrics = []
    rational_residue_ranks = []
    balanced_residue_ranks = []
    hankel_values = []
    raw_orders = []

    for mode_idx in range(num_modes):
        q_col = q_modes[:, mode_idx]

        fit = fit_shared_denominator_vector_rational(
            q_col, num_freqs, value_dim, order, omegas
        )
        poles, residues = decompose_rp_fit_into_first_order(
            fit, q_col, num_freqs, value_dim, omegas
        )
        rational_poles[mode_idx] = poles
        rational_residues[mode_idx] = residues
        rational_residue_ranks.append(
            [
                int(
                    np.linalg.matrix_rank(
                        residues[:, pole_idx].reshape(d, d).T
                    )
                )
                for pole_idx in range(order)
            ]
        )

        lifted = exact_lift_realization(realizations, alpha[:, mode_idx])
        raw_orders.append(lifted.a.shape[0])
        lifted_samples = lifted.frequency_response(omegas)
        lifted_vector = vectorize_transfer_samples(lifted_samples)
        lift_error = np.linalg.norm(lifted_vector - q_col) / max(
            np.linalg.norm(q_col), 1e-15
        )
        if lift_error > 1e-7:
            raise RuntimeError(
                f"Exact lift for PCA mode {mode_idx} failed: {lift_error:.3e}."
            )

        reduced, hsv, bound = square_root_balanced_truncate(
            lifted.a, lifted.b, lifted.c, lifted.d, balanced_order
        )
        reduced_poles, residues, direct = state_space_to_simple_iir_terms(
            reduced
        )
        if reduced_poles.size != balanced_order:
            raise RuntimeError(
                "Balanced reduced order differs from the capacity-matched order."
            )
        balanced_poles[mode_idx] = reduced_poles

        reduced_samples = reduced.frequency_response(omegas)
        zinv = np.exp(-1j * omegas)
        pole_basis = 1.0 / (
            1.0 - zinv[:, None] * reduced_poles[None, :]
        )
        reduced_values = reduced_samples.transpose(0, 2, 1).reshape(
            num_freqs, value_dim
        )
        reconstructed_values = direct[None, :] + pole_basis @ residues.T
        partial_fraction_error = np.linalg.norm(
            reconstructed_values - reduced_values
        ) / max(np.linalg.norm(reduced_values), 1e-15)
        if partial_fraction_error > 1e-6:
            raise RuntimeError(
                "The balanced model could not be represented accurately by "
                f"K simple-pole IIR terms: relative error={partial_fraction_error:.3e}. "
                "Repeated or ill-conditioned poles may require a block/Jordan reservoir."
            )
        balanced_residues[mode_idx] = residues
        balanced_direct[mode_idx] = direct
        balanced_residue_ranks.append(
            [
                int(
                    np.linalg.matrix_rank(
                        residues[:, pole_idx].reshape(d, d).T,
                        tol=1e-7
                        * max(
                            np.linalg.norm(
                                residues[:, pole_idx].reshape(d, d).T,
                                ord=2,
                            ),
                            1e-15,
                        ),
                    )
                )
                for pole_idx in range(balanced_order)
            ]
        )
        partial_fraction_errors.append(float(partial_fraction_error))
        bounds.append(bound)
        hankel_values.append(hsv)

        sample_error = np.max(
            [
                np.linalg.svd(lifted_samples[i] - reduced_samples[i], compute_uv=False)[0]
                for i in range(num_freqs)
            ]
        )
        sampled_errors.append(float(sample_error))

        # Compare both approximations with the same exact lifted PCA mode.
        # The PCA grid reveals the objective optimized by the current fit;
        # a much denser grid tests behavior between those fitting samples.
        zero_direct = np.zeros(value_dim, dtype=np.complex128)
        rational_on_pca_grid = evaluate_iir_response(
            rational_poles[mode_idx],
            rational_residues[mode_idx],
            zero_direct,
            omegas,
            d,
        )
        balanced_on_pca_grid = evaluate_iir_response(
            balanced_poles[mode_idx],
            balanced_residues[mode_idx],
            balanced_direct[mode_idx],
            omegas,
            d,
        )
        rational_pca_grid_metrics.append(
            transfer_approximation_metrics(lifted_samples, rational_on_pca_grid)
        )
        balanced_pca_grid_metrics.append(
            transfer_approximation_metrics(lifted_samples, balanced_on_pca_grid)
        )

        dense_omegas = np.linspace(
            0.0, np.pi, comparison_num_freqs, endpoint=True
        )
        lifted_dense = lifted.frequency_response(dense_omegas)
        rational_dense = evaluate_iir_response(
            rational_poles[mode_idx],
            rational_residues[mode_idx],
            zero_direct,
            dense_omegas,
            d,
        )
        balanced_dense = reduced.frequency_response(dense_omegas)
        rational_dense_grid_metrics.append(
            transfer_approximation_metrics(lifted_dense, rational_dense)
        )
        balanced_dense_grid_metrics.append(
            transfer_approximation_metrics(lifted_dense, balanced_dense)
        )

    # Disable additional pole/input scaling so the experiment compares the
    # actual two configuration procedures.  Both reservoirs use identity.
    rational_esn = ConfiguredWeightsESN(
        rational_poles,
        rational_residues,
        d_out=d,
        d_in=d,
        activation="identity",
        spectral_radius=1.0,
        input_scale=1.0,
    )
    balanced_esn = RankFactoredIIRReservoir(
        balanced_poles,
        balanced_residues,
        d=d,
    )
    diagnostics = {
        "pca_eigenvalues": pca_eigenvalues,
        "balanced_hankel_singular_values": hankel_values,
        "balanced_error_bounds": np.asarray(bounds),
        "balanced_sampled_hinf_errors": np.asarray(sampled_errors),
        "balanced_partial_fraction_errors": np.asarray(
            partial_fraction_errors
        ),
        "rational_pca_grid_metrics": rational_pca_grid_metrics,
        "balanced_pca_grid_metrics": balanced_pca_grid_metrics,
        "rational_dense_grid_metrics": rational_dense_grid_metrics,
        "balanced_dense_grid_metrics": balanced_dense_grid_metrics,
        "comparison_num_freqs": int(comparison_num_freqs),
        "rational_residue_ranks": np.asarray(rational_residue_ranks),
        "balanced_residue_ranks": np.asarray(balanced_residue_ranks),
        "rational_effective_state_count": int(
            np.sum(rational_residue_ranks)
        ),
        "balanced_effective_state_count": int(
            np.sum(balanced_residue_ranks)
        ),
        "balanced_direct_terms": balanced_direct,
        "lifted_raw_orders": np.asarray(raw_orders),
        "rational_feature_dim": num_modes * order * d + d,
        "balanced_feature_dim": balanced_esn.state_dim + d,
        "balanced_order_per_mode": int(balanced_order),
        "current_state_budget": int(num_modes * order * d),
        "balanced_state_budget": int(balanced_esn.state_dim),
    }
    return rational_esn, balanced_esn, diagnostics


def collect_linear_wesn_features(
    esn: ConfiguredWeightsESN | RankFactoredIIRReservoir,
    history: np.ndarray,
    window_length: int = 1,
) -> np.ndarray:
    """Collect reservoir states and the same direct-input skip for both methods."""
    if isinstance(esn, RankFactoredIIRReservoir):
        states = esn.collect_states(history)
    else:
        states = collect_esn_states_per_tile(esn, history)
    if window_length <= 1:
        windowed = history.astype(np.complex128)
    else:
        t_len, num_tiles, d = history.shape
        windowed = np.empty(
            (t_len, num_tiles, window_length * d), dtype=np.complex128
        )
        for time_idx in range(t_len):
            start = max(0, time_idx - window_length + 1)
            window = history[start : time_idx + 1]
            if window.shape[0] < window_length:
                padding = np.repeat(
                    window[0:1], window_length - window.shape[0], axis=0
                )
                window = np.concatenate([padding, window], axis=0)
            windowed[time_idx] = window.transpose(1, 0, 2).reshape(
                num_tiles, -1
            )
    return np.concatenate([states, windowed], axis=-1)


def ridge_predict_next(
    features: np.ndarray,
    observed_history: np.ndarray,
    regularization: float,
) -> np.ndarray:
    z_train = features[:-1].reshape(-1, features.shape[-1])
    y_train = observed_history[1:].reshape(-1, observed_history.shape[-1])
    z_test = features[-1]
    gram = z_train.conj().T @ z_train + regularization * np.eye(
        z_train.shape[1], dtype=np.complex128
    )
    rhs = z_train.conj().T @ y_train
    try:
        readout = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        readout = np.linalg.pinv(gram) @ rhs
    return z_test @ readout


def evaluate_one_snr(
    h_clean: np.ndarray,
    snr_db: float,
    history_len: int,
    ar_order: int,
    num_modes: int,
    order: int,
    num_freqs: int,
    comparison_num_freqs: int,
    offline_ratio: float,
    readout_reg: float,
    seed: int,
) -> tuple[float, float, dict]:
    rng = np.random.default_rng(seed)
    h_noisy, noise_variance = add_complex_awgn(h_clean, snr_db, rng)
    clean_tiles = channels_to_tiles(h_clean)
    noisy_tiles = channels_to_tiles(h_noisy)
    t_dec, _, d = clean_tiles.shape

    offline_len = int(np.floor(t_dec * offline_ratio))
    offline_len = max(history_len + 1, min(offline_len, t_dec - history_len - 1))
    if offline_len <= history_len or t_dec - offline_len <= history_len:
        raise ValueError(
            "Not enough slots for the requested offline split and history length."
        )

    noisy_offline = noisy_tiles[:offline_len]
    noisy_online = noisy_tiles[offline_len:]
    clean_online = clean_tiles[offline_len:]
    r_diag = np.full(d, noise_variance, dtype=np.float64)

    realizations, sampled_vectors, omegas = collect_transfer_ensemble(
        noisy_offline, ar_order, history_len, r_diag, num_freqs
    )
    rational_esn, balanced_esn, diagnostics = build_shared_configurations(
        realizations,
        sampled_vectors,
        omegas,
        num_modes,
        order,
        d,
        comparison_num_freqs=comparison_num_freqs,
    )

    numerators = {"rational": 0.0, "balanced": 0.0}
    denominator = 0.0
    for start in range(noisy_online.shape[0] - history_len):
        observed_history = noisy_online[start : start + history_len]
        truth = clean_online[start + history_len]
        rational_features = collect_linear_wesn_features(
            rational_esn, observed_history, window_length=ar_order
        )
        balanced_features = collect_linear_wesn_features(
            balanced_esn, observed_history, window_length=ar_order
        )
        rational_prediction = ridge_predict_next(
            rational_features, observed_history, readout_reg
        )
        balanced_prediction = ridge_predict_next(
            balanced_features, observed_history, readout_reg
        )
        numerators["rational"] += float(
            np.sum(np.abs(rational_prediction - truth) ** 2)
        )
        numerators["balanced"] += float(
            np.sum(np.abs(balanced_prediction - truth) ** 2)
        )
        denominator += float(np.sum(np.abs(truth) ** 2))

    return (
        numerators["rational"] / max(denominator, 1e-15),
        numerators["balanced"] / max(denominator, 1e-15),
        diagnostics,
    )


def evaluate_production_sequence(
    clean: np.ndarray,
    observed: np.ndarray,
    error_variance: np.ndarray,
    history_len: int,
    ar_order: int,
    num_modes: int,
    order: int,
    num_freqs: int,
    comparison_num_freqs: int,
    offline_ratio: float,
    readout_reg: float,
    num_rx_ues: int,
    num_tx_ues: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Match production's within-drop split and full-tensor NMSE aggregation."""
    num_slots = clean.shape[0]
    offline_count = int(np.floor(num_slots * offline_ratio))
    offline_count = max(1, min(offline_count, num_slots - 1))
    configuration_history_len = min(max(3, ar_order + 1), offline_count)
    # Within-drop production sets rc_predictor.history_len to the number of
    # offline cycles before constructing and freezing the link predictors.
    online_history_len = offline_count
    if offline_count < max(
        configuration_history_len + 1,
        num_modes + configuration_history_len,
    ):
        raise ValueError(
            f"Only {offline_count} offline samples are available; reduce M/history "
            "or provide a longer slot sequence."
        )
    target_indices = np.arange(offline_count + 1, num_slots)
    if target_indices.size == 0:
        raise ValueError("No online targets remain after the production split.")

    rational_predictions = np.zeros_like(clean[target_indices])
    balanced_predictions = np.zeros_like(clean[target_indices])
    link_diagnostics: list[dict] = []

    for rx_node in range(num_rx_ues + 1):
        rx_slice = _node_antenna_slice(rx_node)
        for tx_node in range(num_tx_ues + 1):
            tx_slice = _node_antenna_slice(tx_node)
            observed_link = observed[:, rx_slice, tx_slice]
            clean_link = clean[:, rx_slice, tx_slice]
            error_link = error_variance[:, rx_slice, tx_slice]
            observed_tiles = channels_to_tiles(observed_link)
            _, _, d = observed_tiles.shape
            r_diag = np.maximum(
                np.mean(error_link, axis=(0, 3, 4)).reshape(-1), 1e-12
            )

            realizations, sampled_vectors, omegas = collect_transfer_ensemble(
                observed_tiles[:offline_count],
                ar_order,
                configuration_history_len,
                r_diag,
                num_freqs,
            )
            rational_esn, balanced_esn, diagnostics = build_shared_configurations(
                realizations,
                sampled_vectors,
                omegas,
                num_modes,
                order,
                d,
                comparison_num_freqs=comparison_num_freqs,
            )
            diagnostics["rx_node"] = rx_node
            diagnostics["tx_node"] = tx_node
            link_diagnostics.append(diagnostics)

            # Match the production predictor's persistent recurrence. The
            # first online call replays its available history from zero; each
            # subsequent overlapping history advances that state once.
            rollout_start = int(target_indices[0] - online_history_len)
            rollout_observations = observed_tiles[rollout_start:]
            rational_rollout_features = collect_linear_wesn_features(
                rational_esn,
                rollout_observations,
                window_length=ar_order,
            )
            balanced_rollout_features = collect_linear_wesn_features(
                balanced_esn,
                rollout_observations,
                window_length=ar_order,
            )

            for output_idx, target_idx in enumerate(target_indices):
                history_start = target_idx - online_history_len
                if history_start < 0:
                    raise RuntimeError("Insufficient history for online target.")
                history = observed_tiles[history_start:target_idx]
                feature_start = history_start - rollout_start
                feature_stop = target_idx - rollout_start
                rational_features = rational_rollout_features[
                    feature_start:feature_stop
                ]
                balanced_features = balanced_rollout_features[
                    feature_start:feature_stop
                ]
                rational_link_prediction = ridge_predict_next(
                    rational_features, history, readout_reg
                )
                balanced_link_prediction = ridge_predict_next(
                    balanced_features, history, readout_reg
                )
                rx_count = rx_slice.stop - rx_slice.start
                tx_count = tx_slice.stop - tx_slice.start
                num_symbols = clean_link.shape[-2]
                num_subcarriers = clean_link.shape[-1]
                rational_predictions[
                    output_idx, rx_slice, tx_slice
                ] = rational_link_prediction.reshape(
                    num_symbols, num_subcarriers, rx_count, tx_count
                ).transpose(2, 3, 0, 1)
                balanced_predictions[
                    output_idx, rx_slice, tx_slice
                ] = balanced_link_prediction.reshape(
                    num_symbols, num_subcarriers, rx_count, tx_count
                ).transpose(2, 3, 0, 1)

    truth = clean[target_indices]
    rational_nmse = np.mean(
        np.abs(rational_predictions - truth) ** 2,
        axis=(1, 2, 3, 4),
    ) / np.maximum(np.mean(np.abs(truth) ** 2, axis=(1, 2, 3, 4)), 1e-15)
    balanced_nmse = np.mean(
        np.abs(balanced_predictions - truth) ** 2,
        axis=(1, 2, 3, 4),
    ) / np.maximum(np.mean(np.abs(truth) ** 2, axis=(1, 2, 3, 4)), 1e-15)
    return rational_nmse, balanced_nmse, link_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current rational-fit and balanced-truncation linear "
            "WESN configuration on saved ns-3 channels."
        )
    )
    parser.add_argument("--ns3-root", default="ns3")
    parser.add_argument("--mobility", default="higher_mobility")
    parser.add_argument("--drop-idx", type=int, default=1)
    parser.add_argument("--start-slot", type=int, default=33)
    parser.add_argument("--end-slot", type=int, default=97)
    parser.add_argument("--feedback-delay", type=int, default=4)
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--ar-order", type=int, default=2)
    parser.add_argument("--m", type=int, default=3, help="PCA modes retained")
    parser.add_argument("--k", type=int, default=4, help="Current rational-fit poles per PCA mode")
    parser.add_argument("--num-freqs", type=int, default=64)
    parser.add_argument(
        "--comparison-num-freqs",
        type=int,
        default=2048,
        help="Dense temporal-frequency grid used only to compare approximation errors.",
    )
    parser.add_argument("--rx-ant", type=int, default=2)
    parser.add_argument("--tx-ant", type=int, default=2)
    parser.add_argument("--num-rx-ues", type=int, default=4)
    parser.add_argument("--num-tx-ues", type=int, default=2)
    parser.add_argument("--snr-start", type=int, default=5)
    parser.add_argument("--snr-stop", type=int, default=15)
    parser.add_argument("--snr-step", type=int, default=5)
    parser.add_argument("--offline-ratio", type=float, default=0.5)
    parser.add_argument("--readout-reg", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--synthetic-awgn",
        action="store_true",
        help=(
            "Use the legacy raw-Hdm single-link synthetic-AWGN experiment. "
            "By default the script uses production-normalized truth and saved "
            "LMMSE estimates over the full selected MD-MIMO tensor."
        ),
    )
    parser.add_argument(
        "--output",
        default="results/rational_vs_balanced_wesn_ns3.pdf",
    )
    args = parser.parse_args()

    if not (0.0 < args.offline_ratio < 1.0):
        raise ValueError("--offline-ratio must lie strictly between zero and one.")
    if args.m < 1 or args.k < 1:
        raise ValueError("--m and --k must both be positive.")

    selected_slots = np.arange(
        args.start_slot,
        args.end_slot + 1,
        args.feedback_delay,
        dtype=int,
    )
    if not args.synthetic_awgn:
        clean, observed, error_variance = load_production_channel_sequence(
            Path(args.ns3_root),
            args.mobility,
            args.drop_idx,
            selected_slots,
            args.num_rx_ues,
            args.num_tx_ues,
        )
        estimate_nmse = float(
            np.sum(np.abs(observed - clean) ** 2)
            / max(np.sum(np.abs(clean) ** 2), 1e-15)
        )
        print(
            f"Loaded {selected_slots.size} production slots {selected_slots.tolist()}; "
            f"tensor shape={clean.shape}; saved-estimate NMSE={estimate_nmse:.3e}; "
            f"M={args.m}, current K={args.k}, activation=identity."
        )
        rational_nmse, balanced_nmse, diagnostics = evaluate_production_sequence(
            clean,
            observed,
            error_variance,
            args.history_len,
            args.ar_order,
            args.m,
            args.k,
            args.num_freqs,
            args.comparison_num_freqs,
            args.offline_ratio,
            args.readout_reg,
            args.num_rx_ues,
            args.num_tx_ues,
        )
        rational_hinf = np.asarray(
            [
                metric["hinf_relative"]
                for link in diagnostics
                for metric in link["rational_dense_grid_metrics"]
            ]
        )
        balanced_hinf = np.asarray(
            [
                metric["hinf_relative"]
                for link in diagnostics
                for metric in link["balanced_dense_grid_metrics"]
            ]
        )
        rational_l2 = np.asarray(
            [
                metric["l2_relative"]
                for link in diagnostics
                for metric in link["rational_dense_grid_metrics"]
            ]
        )
        balanced_l2 = np.asarray(
            [
                metric["l2_relative"]
                for link in diagnostics
                for metric in link["balanced_dense_grid_metrics"]
            ]
        )
        print(
            "Full-tensor per-target NMSE\n  rational: "
            + np.array2string(rational_nmse, precision=5)
            + "\n  balanced: "
            + np.array2string(balanced_nmse, precision=5)
        )
        print(
            f"Mean NMSE: rational={np.mean(rational_nmse):.6e}, "
            f"balanced={np.mean(balanced_nmse):.6e}"
        )
        print(
            "Approximation of exact lifted PCA components across all links: "
            f"relative Hinf mean rational={np.mean(rational_hinf):.3e}, "
            f"balanced={np.mean(balanced_hinf):.3e}; relative L2 mean "
            f"rational={np.mean(rational_l2):.3e}, "
            f"balanced={np.mean(balanced_l2):.3e}."
        )

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
        for values, label in (
            (rational_nmse, "Current rational-fit WESN"),
            (balanced_nmse, "Balanced-truncation WESN"),
        ):
            sorted_values = np.sort(values)
            cdf = np.arange(1, sorted_values.size + 1) / sorted_values.size
            axes[0].plot(sorted_values, cdf, label=label)
        axes[0].set_xlabel("Channel prediction NMSE")
        axes[0].set_ylabel("CDF")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(frameon=False)
        axes[1].bar(
            np.arange(2) - 0.18,
            [np.mean(rational_hinf), np.mean(balanced_hinf)],
            width=0.36,
            label=r"Relative $H_\infty$",
        )
        axes[1].bar(
            np.arange(2) + 0.18,
            [np.mean(rational_l2), np.mean(balanced_l2)],
            width=0.36,
            label=r"Relative $L_2$",
        )
        axes[1].set_xticks(np.arange(2), ["Rational", "Balanced"])
        axes[1].set_ylabel("Mean relative PCA-component error")
        axes[1].set_yscale("log")
        axes[1].grid(True, axis="y", alpha=0.3)
        axes[1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output)
        plt.close(fig)
        data_output = output.with_suffix(".npz")
        np.savez(
            data_output,
            selected_slots=selected_slots,
            rational_nmse=rational_nmse,
            balanced_nmse=balanced_nmse,
            rational_relative_hinf=rational_hinf,
            balanced_relative_hinf=balanced_hinf,
            rational_relative_l2=rational_l2,
            balanced_relative_l2=balanced_l2,
            estimate_nmse=np.asarray(estimate_nmse),
            m=np.asarray(args.m),
            current_k=np.asarray(args.k),
        )
        print(f"Saved plot to {output}")
        print(f"Saved numerical results to {data_output}")
        return

    h_clean, selected_slots = load_clean_p2p_channels(
        ns3_folder=Path(args.ns3_root),
        drop_idx=args.drop_idx,
        mobility=args.mobility,
        start_slot=args.start_slot,
        end_slot=args.end_slot,
        feedback_delay=args.feedback_delay,
        rx_ant=args.rx_ant,
        tx_ant=args.tx_ant,
    )
    print(
        f"Loaded {selected_slots.size} decimated slots; channel shape={h_clean.shape}; "
        f"M={args.m}, K={args.k}, activation=identity."
    )

    snrs = np.arange(args.snr_start, args.snr_stop + 1, args.snr_step)
    rational_nmse = []
    balanced_nmse = []
    bound_rows = []
    sampled_error_rows = []
    rational_dense_hinf_rows = []
    balanced_dense_hinf_rows = []
    rational_dense_l2_rows = []
    balanced_dense_l2_rows = []
    rational_pca_hinf_rows = []
    balanced_pca_hinf_rows = []

    for snr_db in snrs:
        nmse_rational, nmse_balanced, diagnostics = evaluate_one_snr(
            h_clean=h_clean,
            snr_db=float(snr_db),
            history_len=args.history_len,
            ar_order=args.ar_order,
            num_modes=args.m,
            order=args.k,
            num_freqs=args.num_freqs,
            comparison_num_freqs=args.comparison_num_freqs,
            offline_ratio=args.offline_ratio,
            readout_reg=args.readout_reg,
            seed=args.seed,
        )
        rational_nmse.append(nmse_rational)
        balanced_nmse.append(nmse_balanced)
        bound_rows.append(diagnostics["balanced_error_bounds"])
        sampled_error_rows.append(diagnostics["balanced_sampled_hinf_errors"])
        rational_dense_hinf_rows.append(
            [
                metrics["hinf_relative"]
                for metrics in diagnostics["rational_dense_grid_metrics"]
            ]
        )
        balanced_dense_hinf_rows.append(
            [
                metrics["hinf_relative"]
                for metrics in diagnostics["balanced_dense_grid_metrics"]
            ]
        )
        rational_dense_l2_rows.append(
            [
                metrics["l2_relative"]
                for metrics in diagnostics["rational_dense_grid_metrics"]
            ]
        )
        balanced_dense_l2_rows.append(
            [
                metrics["l2_relative"]
                for metrics in diagnostics["balanced_dense_grid_metrics"]
            ]
        )
        rational_pca_hinf_rows.append(
            [
                metrics["hinf_relative"]
                for metrics in diagnostics["rational_pca_grid_metrics"]
            ]
        )
        balanced_pca_hinf_rows.append(
            [
                metrics["hinf_relative"]
                for metrics in diagnostics["balanced_pca_grid_metrics"]
            ]
        )
        print(
            f"SNR={snr_db:>3} dB | rational NMSE={nmse_rational:.6e} | "
            f"balanced NMSE={nmse_balanced:.6e}"
        )
        print(
            "  BT mode bounds: "
            + ", ".join(
                f"m={idx + 1}: measured-grid={measured:.3e}, bound={bound:.3e}"
                for idx, (measured, bound) in enumerate(
                    zip(
                        diagnostics["balanced_sampled_hinf_errors"],
                        diagnostics["balanced_error_bounds"],
                    )
                )
            )
        )
        print(
            "  Approximation of exact lifted Q_m(z) "
            f"({args.comparison_num_freqs}-point dense grid):"
        )
        print(
            "  Capacity match: "
            f"current rational states={diagnostics['current_state_budget']}, "
            f"balanced states={diagnostics['balanced_state_budget']}; "
            f"current K={args.k} full-rank matrix residues, balanced order/component="
            f"{diagnostics['balanced_order_per_mode']}."
        )
        for mode_idx in range(args.m):
            rational_pca = diagnostics["rational_pca_grid_metrics"][mode_idx]
            balanced_pca = diagnostics["balanced_pca_grid_metrics"][mode_idx]
            rational_dense = diagnostics["rational_dense_grid_metrics"][mode_idx]
            balanced_dense = diagnostics["balanced_dense_grid_metrics"][mode_idx]
            print(
                f"    m={mode_idx + 1}: relative Hinf "
                f"PCA-grid rational={rational_pca['hinf_relative']:.3e}, "
                f"BT={balanced_pca['hinf_relative']:.3e}; dense rational="
                f"{rational_dense['hinf_relative']:.3e}, "
                f"BT={balanced_dense['hinf_relative']:.3e}; relative L2 "
                f"dense rational={rational_dense['l2_relative']:.3e}, "
                f"BT={balanced_dense['l2_relative']:.3e}"
            )

    rational_nmse = np.asarray(rational_nmse)
    balanced_nmse = np.asarray(balanced_nmse)
    rational_dense_hinf_array = np.asarray(rational_dense_hinf_rows)
    balanced_dense_hinf_array = np.asarray(balanced_dense_hinf_rows)
    rational_dense_l2_array = np.asarray(rational_dense_l2_rows)
    balanced_dense_l2_array = np.asarray(balanced_dense_l2_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    nmse_ax, approximation_ax = axes
    nmse_ax.semilogy(
        snrs,
        rational_nmse,
        marker="o",
        markerfacecolor="white",
        label="Current rational-fit WESN",
    )
    nmse_ax.semilogy(
        snrs,
        balanced_nmse,
        marker="s",
        markerfacecolor="white",
        label="Balanced-truncation WESN",
    )
    nmse_ax.set_xlabel("SNR (dB)")
    nmse_ax.set_ylabel("Channel prediction NMSE")
    nmse_ax.grid(True, which="both", alpha=0.3)
    nmse_ax.legend(frameon=False)

    approximation_ax.semilogy(
        snrs,
        np.mean(rational_dense_hinf_array, axis=1),
        marker="o",
        markerfacecolor="white",
        label=r"Rational: relative $H_\infty$",
    )
    approximation_ax.semilogy(
        snrs,
        np.mean(balanced_dense_hinf_array, axis=1),
        marker="s",
        markerfacecolor="white",
        label=r"Balanced: relative $H_\infty$",
    )
    approximation_ax.semilogy(
        snrs,
        np.mean(rational_dense_l2_array, axis=1),
        marker="o",
        linestyle="--",
        markerfacecolor="white",
        label=r"Rational: relative $L_2$",
    )
    approximation_ax.semilogy(
        snrs,
        np.mean(balanced_dense_l2_array, axis=1),
        marker="s",
        linestyle="--",
        markerfacecolor="white",
        label=r"Balanced: relative $L_2$",
    )
    approximation_ax.set_xlabel("SNR (dB)")
    approximation_ax.set_ylabel(r"Mean relative error across $M$ modes")
    approximation_ax.grid(True, which="both", alpha=0.3)
    approximation_ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)

    data_output = output.with_suffix(".npz")
    np.savez(
        data_output,
        snr_db=snrs,
        rational_nmse=rational_nmse,
        balanced_nmse=balanced_nmse,
        balanced_error_bounds=np.asarray(bound_rows),
        balanced_sampled_hinf_errors=np.asarray(sampled_error_rows),
        rational_pca_grid_relative_hinf=np.asarray(rational_pca_hinf_rows),
        balanced_pca_grid_relative_hinf=np.asarray(balanced_pca_hinf_rows),
        rational_dense_grid_relative_hinf=rational_dense_hinf_array,
        balanced_dense_grid_relative_hinf=balanced_dense_hinf_array,
        rational_dense_grid_relative_l2=rational_dense_l2_array,
        balanced_dense_grid_relative_l2=balanced_dense_l2_array,
        comparison_num_freqs=np.asarray(args.comparison_num_freqs),
        m=np.asarray(args.m),
        k=np.asarray(args.k),
        balanced_order_per_mode=np.asarray(
            args.k * args.rx_ant * args.tx_ant
        ),
        selected_slots=selected_slots,
    )
    print(f"Saved plot to {output}")
    print(f"Saved numerical results to {data_output}")


if __name__ == "__main__":
    main()
