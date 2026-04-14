
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Basic helpers
# ============================================================

def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def project_to_stable_matrix(F: np.ndarray, max_radius: float = 0.99) -> np.ndarray:
    eigvals = np.linalg.eigvals(F)
    rho = np.max(np.abs(eigvals))
    if rho >= max_radius:
        F = F * (max_radius / (rho + 1e-15))
    return F


# ============================================================
# Steady-state Kalman predictor ingredients
# ============================================================

def solve_discrete_riccati_steady_state(
    F: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    max_iter: int = 10000,
    tol: float = 1e-12,
):
    d = F.shape[0]
    P_minus = np.eye(d)

    for _ in range(max_iter):
        S = P_minus + R
        K = P_minus @ np.linalg.inv(S)
        P_plus = P_minus - K @ P_minus
        P_minus_next = F @ P_plus @ F.T + Q

        err = np.linalg.norm(P_minus_next - P_minus, ord="fro")
        P_minus = P_minus_next
        if err < tol:
            break

    S = P_minus + R
    K = P_minus @ np.linalg.inv(S)
    return P_minus, K


def solve_stationary_state_covariance(
    F: np.ndarray,
    Q: np.ndarray,
    max_iter: int = 10000,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Solve P = F P F^T + Q by fixed-point iteration (stable F assumed).
    """
    d = F.shape[0]
    P = np.eye(d)
    for _ in range(max_iter):
        P_next = F @ P @ F.T + Q
        P_next = symmetrize(P_next)
        if np.linalg.norm(P_next - P, ord="fro") < tol:
            return P_next
        P = P_next
    return P


def normalize_qr_for_unit_expected_state_norm(
    F: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    target_power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Scale (Q, R) by the same factor so that the stationary state power
    E[||x_t||^2] = trace(P) equals target_power.
    """
    P_stat = solve_stationary_state_covariance(F, Q)
    state_power = float(np.real(np.trace(P_stat)))
    scale = float(target_power / max(state_power, 1e-15))
    Q_scaled = Q * scale
    R_scaled = R * scale
    return Q_scaled, R_scaled, scale, state_power


def scale_matrix_to_spectral_radius(F: np.ndarray, target_radius: float) -> np.ndarray:
    rho = float(np.max(np.abs(np.linalg.eigvals(F))))
    if rho < 1e-15:
        raise ValueError("Cannot scale a near-zero matrix to a target spectral radius.")
    return F * (target_radius / rho)


def enforce_psd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = symmetrize(M)
    evals, evecs = np.linalg.eigh(M)
    evals = np.maximum(evals, eps)
    return evecs @ np.diag(evals) @ evecs.T


def build_unit_power_model_from_template(
    F_template: np.ndarray,
    rho_mobility: float,
    target_power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = F_template.shape[0]
    C_target = (target_power / d) * np.eye(d)
    F_model = scale_matrix_to_spectral_radius(F_template, rho_mobility)
    Q_model = C_target - F_model @ C_target @ F_model.T
    Q_model = enforce_psd(Q_model)
    return F_model, Q_model, C_target


def generate_state_space_data(
    T: int,
    F: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    rng: np.random.Generator,
):
    d = F.shape[0]
    x = np.zeros((T, d))
    y = np.zeros((T, d))

    P_stat = solve_stationary_state_covariance(F, Q)
    x_prev = rng.multivariate_normal(np.zeros(d), P_stat)

    for t in range(T):
        w_t = rng.multivariate_normal(np.zeros(d), Q)
        x_t = F @ x_prev + w_t

        n_t = rng.multivariate_normal(np.zeros(d), R)
        y_t = x_t + n_t

        x[t] = x_t
        y[t] = y_t
        x_prev = x_t

    return x, y


def kalman_filter_identity_H(y: np.ndarray, F: np.ndarray, Q: np.ndarray, R: np.ndarray):
    T, d = y.shape

    x_pred = np.zeros((T, d))
    P_pred = np.zeros((T, d, d))
    x_filt = np.zeros((T, d))
    P_filt = np.zeros((T, d, d))
    K_hist = np.zeros((T, d, d))

    x_plus_prev = np.zeros(d)
    P_plus_prev = np.eye(d)

    for t in range(T):
        x_minus = F @ x_plus_prev
        P_minus = F @ P_plus_prev @ F.T + Q
        P_minus = symmetrize(P_minus)

        S = P_minus + R
        K_t = P_minus @ np.linalg.inv(S)
        innovation = y[t] - x_minus
        x_plus = x_minus + K_t @ innovation
        P_plus = (np.eye(d) - K_t) @ P_minus
        P_plus = symmetrize(P_plus)

        x_pred[t] = x_minus
        P_pred[t] = P_minus
        x_filt[t] = x_plus
        P_filt[t] = P_plus
        K_hist[t] = K_t

        x_plus_prev = x_plus
        P_plus_prev = P_plus

    return x_pred, P_pred, x_filt, P_filt, K_hist


def rts_smoother_with_lag_cov(y: np.ndarray, F: np.ndarray, Q: np.ndarray, R: np.ndarray):
    x_pred, P_pred, x_filt, P_filt, K_hist = kalman_filter_identity_H(y, F, Q, R)

    T, d = y.shape

    x_smooth = np.zeros((T, d))
    P_smooth = np.zeros((T, d, d))
    J = np.zeros((max(T - 1, 1), d, d))

    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for t in range(T - 2, -1, -1):
        J_t = P_filt[t] @ F.T @ np.linalg.inv(P_pred[t + 1])
        J[t] = J_t

        x_smooth[t] = x_filt[t] + J_t @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + J_t @ (P_smooth[t + 1] - P_pred[t + 1]) @ J_t.T
        P_smooth[t] = symmetrize(P_smooth[t])

    P_lag = np.zeros((T, d, d))
    if T >= 2:
        P_lag[T - 1] = (np.eye(d) - K_hist[T - 1]) @ F @ P_filt[T - 2]

        for t in range(T - 2, 0, -1):
            P_lag[t] = (
                P_filt[t] @ J[t - 1].T
                + J[t] @ (P_lag[t + 1] - F @ P_filt[t]) @ J[t - 1].T
            )
            P_lag[t] = symmetrize(P_lag[t])

    return x_smooth, P_smooth, P_lag


def estimate_F_init_from_covariances(y: np.ndarray, R: np.ndarray, reg: float = 1e-8) -> np.ndarray:
    y_centered = y - np.mean(y, axis=0, keepdims=True)

    Y1 = y_centered[1:]
    Y0 = y_centered[:-1]

    Gamma_y_1 = (Y1.T @ Y0) / max(len(Y0), 1)
    Gamma_y_0 = (Y0.T @ Y0) / max(len(Y0), 1)

    Gamma_x_0 = symmetrize(Gamma_y_0 - R)
    Gamma_x_0 = Gamma_x_0 + reg * np.eye(Gamma_x_0.shape[0])

    F0 = Gamma_y_1 @ np.linalg.pinv(Gamma_x_0)
    F0 = project_to_stable_matrix(F0, max_radius=0.99)
    return F0


def estimate_F_from_y_em(
    y: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    F_init: np.ndarray | None = None,
    max_em_iters: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    T, d = y.shape

    if T < 2:
        raise ValueError("Need at least T >= 2 samples to estimate F.")

    if F_init is None:
        F = estimate_F_init_from_covariances(y, R)
    else:
        F = F_init.copy()

    F = project_to_stable_matrix(F, max_radius=0.99)

    for _ in range(max_em_iters):
        F_old = F.copy()

        x_smooth, P_smooth, P_lag = rts_smoother_with_lag_cov(y, F, Q, R)

        S10 = np.zeros((d, d))
        S00 = np.zeros((d, d))

        for t in range(1, T):
            Ex_t_xtm1 = P_lag[t] + np.outer(x_smooth[t], x_smooth[t - 1])
            Ex_tm1_xtm1 = P_smooth[t - 1] + np.outer(x_smooth[t - 1], x_smooth[t - 1])

            S10 += Ex_t_xtm1
            S00 += Ex_tm1_xtm1

        F = S10 @ np.linalg.pinv(S00)
        F = project_to_stable_matrix(F, max_radius=0.99)

        delta = np.linalg.norm(F - F_old, ord="fro")
        if delta < tol:
            break

    return F


def select_chunk_F(
    y_hist: np.ndarray,
    F_true: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    use_estimated_F: bool,
    max_em_iters: int = 50,
    em_tol: float = 1e-6,
) -> np.ndarray:
    if not use_estimated_F:
        return F_true.copy()

    F_init = estimate_F_init_from_covariances(y_hist, R)
    F_hat = estimate_F_from_y_em(
        y=y_hist,
        Q=Q,
        R=R,
        F_init=F_init,
        max_em_iters=max_em_iters,
        tol=em_tol,
    )
    return F_hat


def steady_state_predictor_matrices(F_hat: np.ndarray, Q: np.ndarray, R: np.ndarray):
    _, K_ss = solve_discrete_riccati_steady_state(F_hat, Q, R)
    A_p = F_hat - F_hat @ K_ss
    B_p = F_hat @ K_ss
    return A_p, B_p, K_ss


def run_steady_state_predictor(y_hist: np.ndarray, F: np.ndarray, K_ss: np.ndarray) -> np.ndarray:
    """
    Uses y[0],...,y[T_hist-1] to produce one-step-ahead predictions:
        xhat_pred[t] = xhat_{t+1|t}
    """
    T_hist, d = y_hist.shape
    A_p = F - F @ K_ss
    B_p = F @ K_ss

    xhat_pred = np.zeros((T_hist, d))
    s_prev = np.zeros(d)

    for t in range(T_hist):
        s_t = A_p @ s_prev + B_p @ y_hist[t]
        xhat_pred[t] = s_t
        s_prev = s_t

    return xhat_pred


def run_full_kalman_one_step_predictor(
    y_hist: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    Uses measurements y_hist = [y_0, ..., y_{T_hist-1}] to produce
    one-step-ahead predictions:
        xhat_pred[t] = xhat_{t+1|t}
    """
    T_hist, d = y_hist.shape

    xhat_plus = np.zeros(d)
    P_plus = np.eye(d)
    xhat_pred = np.zeros((T_hist, d))

    for t in range(T_hist):
        xhat_minus = F @ xhat_plus
        P_minus = F @ P_plus @ F.T + Q

        S = P_minus + R
        K_t = P_minus @ np.linalg.inv(S)
        innovation = y_hist[t] - xhat_minus
        xhat_plus = xhat_minus + K_t @ innovation
        P_plus = (np.eye(d) - K_t) @ P_minus

        xhat_pred[t] = F @ xhat_plus

    return xhat_pred


def steady_state_predictor_transfer_samples(
    A_p: np.ndarray,
    B_p: np.ndarray,
    num_freqs: int,
    include_dc_to_2pi: bool = False,
):
    d = A_p.shape[0]

    if include_dc_to_2pi:
        omegas = np.linspace(0.0, 2.0 * np.pi, num_freqs, endpoint=False)
    else:
        omegas = np.linspace(0.0, np.pi, num_freqs, endpoint=True)

    H_samps = np.zeros((num_freqs, d, d), dtype=np.complex128)
    I = np.eye(d, dtype=np.complex128)

    for i, w in enumerate(omegas):
        zinv = np.exp(-1j * w)
        H_samps[i] = np.linalg.inv(I - A_p.astype(np.complex128) * zinv) @ B_p.astype(np.complex128)

    return H_samps


def vectorize_transfer_samples(H_samps: np.ndarray) -> np.ndarray:
    num_freqs, d_out, d_in = H_samps.shape
    v_blocks = [H_samps[i].reshape(d_out * d_in, order="F") for i in range(num_freqs)]
    return np.concatenate(v_blocks, axis=0)


def generate_chunkwise_transfer_vectors(
    F_true: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    history_len: int,
    num_chunks: int,
    num_freqs: int,
    use_estimated_F: bool,
    seed: int = 0,
    max_em_iters: int = 50,
    em_tol: float = 1e-6,
):
    rng = np.random.default_rng(seed)

    total_T = num_chunks * (history_len + 1)
    x_all, y_all = generate_state_space_data(total_T, F_true, Q, R, rng)

    chunk_len = history_len + 1

    V_list = []
    F_hats = []
    A_ps = []
    B_ps = []
    H_chunks = []

    for k in range(num_chunks):
        start = k * chunk_len
        end = (k + 1) * chunk_len

        y_chunk = y_all[start:end]
        y_hist = y_chunk[:history_len]

        F_used = select_chunk_F(
            y_hist=y_hist,
            F_true=F_true,
            Q=Q,
            R=R,
            use_estimated_F=use_estimated_F,
            max_em_iters=max_em_iters,
            em_tol=em_tol,
        )

        A_p, B_p, _ = steady_state_predictor_matrices(F_used, Q, R)
        H_samps = steady_state_predictor_transfer_samples(
            A_p=A_p,
            B_p=B_p,
            num_freqs=num_freqs,
            include_dc_to_2pi=False,
        )
        v_k = vectorize_transfer_samples(H_samps)

        V_list.append(v_k)
        F_hats.append(F_used)
        A_ps.append(A_p)
        B_ps.append(B_p)
        H_chunks.append(H_samps)

    V = np.stack(V_list, axis=1)

    return {
        "x_all": x_all,
        "y_all": y_all,
        "V": V,
        "F_hats": np.stack(F_hats, axis=0),
        "A_ps": np.stack(A_ps, axis=0),
        "B_ps": np.stack(B_ps, axis=0),
        "transfer_samples": np.stack(H_chunks, axis=0),
    }


# ============================================================
# Covariance / PCA / transfer-space NMSE
# ============================================================

def estimate_empirical_complex_covariance(V: np.ndarray):
    mean_v = np.mean(V, axis=1, keepdims=True)
    Vc = V - mean_v
    K_vv = (Vc @ Vc.conj().T) / V.shape[1]
    K_vv = 0.5 * (K_vv + K_vv.conj().T)
    return mean_v, Vc, K_vv


def pca_from_covariance(K_vv: np.ndarray):
    evals, evecs = np.linalg.eigh(K_vv)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals, evecs


def reconstruct_with_first_m_modes(
    V: np.ndarray,
    mean_v: np.ndarray,
    Q: np.ndarray,
    m: int,
) -> np.ndarray:
    if m == 0:
        return np.repeat(mean_v, V.shape[1], axis=1)

    Qm = Q[:, :m]
    Vc = V - mean_v
    Vhat = mean_v + Qm @ (Qm.conj().T @ Vc)
    return Vhat


def reconstruct_with_general_basis(
    V: np.ndarray,
    mean_v: np.ndarray,
    F_basis: np.ndarray,
) -> np.ndarray:
    if F_basis.size == 0 or F_basis.shape[1] == 0:
        return np.repeat(mean_v, V.shape[1], axis=1)

    Vc = V - mean_v
    G = F_basis.conj().T @ F_basis
    Ginv = np.linalg.pinv(G)
    Vhat = mean_v + F_basis @ (Ginv @ (F_basis.conj().T @ Vc))
    return Vhat


def nmse_between_matrices(Vhat: np.ndarray, V: np.ndarray) -> float:
    num = np.linalg.norm(Vhat - V, ord="fro") ** 2
    den = np.linalg.norm(V, ord="fro") ** 2
    return float(np.real(num / max(den, 1e-15)))


def compute_reconstruction_nmse_curve(
    V: np.ndarray,
    mean_v: np.ndarray,
    Q: np.ndarray,
    max_m: int | None = None,
):
    Dv = V.shape[0]
    if max_m is None:
        max_m = Dv

    max_m = min(max_m, Dv)
    m_vals = np.arange(0, max_m + 1)
    nmse_vals = np.zeros_like(m_vals, dtype=float)

    for i, m in enumerate(m_vals):
        Vhat = reconstruct_with_first_m_modes(V, mean_v, Q, m)
        nmse_vals[i] = nmse_between_matrices(Vhat, V)

    return m_vals, nmse_vals


# ============================================================
# Rational approximation helpers
# ============================================================

def q_column_to_frequency_matrix(q_col: np.ndarray, num_freqs: int, value_dim: int) -> np.ndarray:
    if q_col.ndim != 1:
        raise ValueError("q_col must be a 1D array.")
    if q_col.size != num_freqs * value_dim:
        raise ValueError(
            f"q_col length {q_col.size} does not match num_freqs*value_dim = {num_freqs*value_dim}."
        )

    Qf = np.zeros((num_freqs, value_dim), dtype=np.complex128)
    for n in range(num_freqs):
        start = n * value_dim
        stop = (n + 1) * value_dim
        Qf[n, :] = q_col[start:stop]
    return Qf


def frequency_matrix_to_q_column(Qf: np.ndarray) -> np.ndarray:
    return Qf.reshape(-1)


def fit_shared_denominator_vector_rational(
    q_col: np.ndarray,
    num_freqs: int,
    value_dim: int,
    degree: int,
    omegas: np.ndarray | None = None,
):
    if degree < 1:
        raise ValueError("degree must be >= 1")

    Qf = q_column_to_frequency_matrix(q_col, num_freqs, value_dim)

    if omegas is None:
        omegas = np.linspace(0.0, np.pi, num_freqs, endpoint=True)

    zinv = np.exp(-1j * omegas)
    Z = np.stack([zinv ** k for k in range(1, degree + 1)], axis=1)

    num_unknowns = degree + value_dim * degree
    num_rows = num_freqs * value_dim

    A_ls = np.zeros((num_rows, num_unknowns), dtype=np.complex128)
    y_ls = np.zeros((num_rows,), dtype=np.complex128)

    row = 0
    for n in range(num_freqs):
        for r in range(value_dim):
            qnr = Qf[n, r]

            A_ls[row, 0:degree] = qnr * Z[n, :]
            num_start = degree + r * degree
            num_stop = degree + (r + 1) * degree
            A_ls[row, num_start:num_stop] = -Z[n, :]
            y_ls[row] = -qnr
            row += 1

    theta, *_ = np.linalg.lstsq(A_ls, y_ls, rcond=None)

    a_den = theta[0:degree]
    b_num = theta[degree:].reshape(value_dim, degree)

    denom = 1.0 + Z @ a_den
    Qf_hat = np.zeros_like(Qf)

    for r in range(value_dim):
        numer = Z @ b_num[r, :]
        Qf_hat[:, r] = numer / denom

    q_hat_col = frequency_matrix_to_q_column(Qf_hat)

    fit_num = np.linalg.norm(q_hat_col - q_col) ** 2
    fit_den = np.linalg.norm(q_col) ** 2
    q_fit_nmse = float(np.real(fit_num / max(fit_den, 1e-15)))

    residual = A_ls @ theta - y_ls
    residual_norm = float(np.linalg.norm(residual) ** 2)

    return {
        "a_den": a_den,
        "b_num": b_num,
        "q_hat_col": q_hat_col,
        "q_fit_nmse": q_fit_nmse,
        "residual_norm": residual_norm,
    }


def fit_first_m_q_columns_with_rational_models(
    Q_eig: np.ndarray,
    num_freqs: int,
    value_dim: int,
    m: int,
    degree: int,
    omegas: np.ndarray | None = None,
):
    Dv, Dq = Q_eig.shape
    m = min(m, Dq)

    Q_rp_raw = np.zeros((Dv, m), dtype=np.complex128)
    fit_nmses = np.zeros((m,), dtype=float)
    fit_info = []

    for j in range(m):
        fit = fit_shared_denominator_vector_rational(
            q_col=Q_eig[:, j],
            num_freqs=num_freqs,
            value_dim=value_dim,
            degree=degree,
            omegas=omegas,
        )
        Q_rp_raw[:, j] = fit["q_hat_col"]
        fit_nmses[j] = fit["q_fit_nmse"]
        fit_info.append(fit)

    return Q_rp_raw, fit_nmses, fit_info


# ============================================================
# First-order decomposition helpers
# ============================================================

def denominator_coeffs_to_poles(a_den: np.ndarray) -> np.ndarray:
    degree = len(a_den)
    coeff_desc = np.concatenate([a_den[::-1], np.array([1.0 + 0.0j])])
    roots_x = np.roots(coeff_desc)

    eps = 1e-12
    poles = np.zeros_like(roots_x, dtype=np.complex128)
    for i, rx in enumerate(roots_x):
        if np.abs(rx) < eps:
            poles[i] = 0.0
        else:
            poles[i] = 1.0 / rx
    return poles


def stabilize_poles(poles: np.ndarray, max_radius: float = 0.98) -> np.ndarray:
    poles_stable = poles.copy().astype(np.complex128)
    for i, p in enumerate(poles_stable):
        mag = np.abs(p)
        if mag >= max_radius:
            if mag < 1e-12:
                poles_stable[i] = 0.0
            else:
                poles_stable[i] = p * (max_radius / mag)
    return poles_stable


def decompose_rp_fit_into_first_order(
    fit_dict: dict,
    num_freqs: int,
    value_dim: int,
    omegas: np.ndarray | None = None,
):
    a_den = fit_dict["a_den"]
    q_rp_col = fit_dict["q_hat_col"]

    if omegas is None:
        omegas = np.linspace(0.0, np.pi, num_freqs, endpoint=True)

    Qf_rp = q_column_to_frequency_matrix(q_rp_col, num_freqs, value_dim)
    poles = denominator_coeffs_to_poles(a_den)
    poles = stabilize_poles(poles, max_radius=0.98)
    num_poles = len(poles)

    zinv = np.exp(-1j * omegas)
    Phi = 1.0 / (1.0 - zinv[:, None] * poles[None, :])

    residues = np.zeros((value_dim, num_poles), dtype=np.complex128)
    Qf_fo = np.zeros_like(Qf_rp)

    for r in range(value_dim):
        c_r, *_ = np.linalg.lstsq(Phi, Qf_rp[:, r], rcond=None)
        residues[r, :] = c_r
        Qf_fo[:, r] = Phi @ c_r

    q_fo_col = frequency_matrix_to_q_column(Qf_fo)
    fo_num = np.linalg.norm(q_fo_col - fit_dict["q_hat_col"]) ** 2
    fo_den = np.linalg.norm(fit_dict["q_hat_col"]) ** 2
    fo_vs_rp_nmse = float(np.real(fo_num / max(fo_den, 1e-15)))

    return {
        "poles": poles,
        "residues": residues,
        "q_fo_col": q_fo_col,
        "fo_vs_rp_nmse": fo_vs_rp_nmse,
    }


def decompose_first_m_rp_columns_to_first_order(
    fit_info: list,
    num_freqs: int,
    value_dim: int,
    omegas: np.ndarray | None = None,
):
    m = len(fit_info)
    Dv = fit_info[0]["q_hat_col"].size if m > 0 else 0

    Q_fo_raw = np.zeros((Dv, m), dtype=np.complex128)
    fo_fit_nmses = np.zeros((m,), dtype=float)
    fo_info = []

    for j, fit in enumerate(fit_info):
        fo = decompose_rp_fit_into_first_order(
            fit_dict=fit,
            num_freqs=num_freqs,
            value_dim=value_dim,
            omegas=omegas,
        )
        Q_fo_raw[:, j] = fo["q_fo_col"]
        fo_fit_nmses[j] = fo["fo_vs_rp_nmse"]
        fo_info.append(fo)

    return Q_fo_raw, fo_fit_nmses, fo_info


def compute_rp_and_fo_basis_reconstruction_nmse_curve(
    V: np.ndarray,
    mean_v: np.ndarray,
    Q_eig: np.ndarray,
    num_freqs: int,
    value_dim: int,
    degree: int,
    max_m: int,
):
    max_m = min(max_m, Q_eig.shape[1])
    m_vals = np.arange(0, max_m + 1)

    nmse_vals_rp_basis = np.zeros((len(m_vals),), dtype=float)
    nmse_vals_fo_basis = np.zeros((len(m_vals),), dtype=float)
    avg_q_fit_nmse_vals = np.zeros((len(m_vals),), dtype=float)
    avg_fo_fit_nmse_vals = np.zeros((len(m_vals),), dtype=float)

    omegas = np.linspace(0.0, np.pi, num_freqs, endpoint=True)

    for i, m in enumerate(m_vals):
        if m == 0:
            Vhat = np.repeat(mean_v, V.shape[1], axis=1)
            nmse_vals_rp_basis[i] = nmse_between_matrices(Vhat, V)
            nmse_vals_fo_basis[i] = nmse_between_matrices(Vhat, V)
            avg_q_fit_nmse_vals[i] = 1.0
            avg_fo_fit_nmse_vals[i] = 1.0
            continue

        Q_rp_raw, fit_nmses, fit_info = fit_first_m_q_columns_with_rational_models(
            Q_eig=Q_eig,
            num_freqs=num_freqs,
            value_dim=value_dim,
            m=m,
            degree=degree,
            omegas=omegas,
        )

        Vhat_rp = reconstruct_with_general_basis(V, mean_v, Q_rp_raw)
        nmse_vals_rp_basis[i] = nmse_between_matrices(Vhat_rp, V)
        avg_q_fit_nmse_vals[i] = float(np.mean(fit_nmses))

        Q_fo_raw, fo_fit_nmses, _ = decompose_first_m_rp_columns_to_first_order(
            fit_info=fit_info,
            num_freqs=num_freqs,
            value_dim=value_dim,
            omegas=omegas,
        )

        Vhat_fo = reconstruct_with_general_basis(V, mean_v, Q_fo_raw)
        nmse_vals_fo_basis[i] = nmse_between_matrices(Vhat_fo, V)
        avg_fo_fit_nmse_vals[i] = float(np.mean(fo_fit_nmses))

    return (
        m_vals,
        nmse_vals_rp_basis,
        nmse_vals_fo_basis,
        avg_q_fit_nmse_vals,
        avg_fo_fit_nmse_vals,
    )


def compute_degree_sweep_with_fo(
    V: np.ndarray,
    mean_v: np.ndarray,
    Q_eig: np.ndarray,
    num_freqs: int,
    value_dim: int,
    m_eval: int,
    degree_vals: np.ndarray,
):
    m_eval = min(m_eval, Q_eig.shape[1])

    v_recon_nmse_vs_degree_rp = np.zeros((len(degree_vals),), dtype=float)
    v_recon_nmse_vs_degree_fo = np.zeros((len(degree_vals),), dtype=float)

    omegas = np.linspace(0.0, np.pi, num_freqs, endpoint=True)

    for i, deg in enumerate(degree_vals):
        Q_rp_raw, _, fit_info = fit_first_m_q_columns_with_rational_models(
            Q_eig=Q_eig,
            num_freqs=num_freqs,
            value_dim=value_dim,
            m=m_eval,
            degree=int(deg),
            omegas=omegas,
        )

        Vhat_rp = reconstruct_with_general_basis(V, mean_v, Q_rp_raw)
        v_recon_nmse_vs_degree_rp[i] = nmse_between_matrices(Vhat_rp, V)

        Q_fo_raw, _, _ = decompose_first_m_rp_columns_to_first_order(
            fit_info=fit_info,
            num_freqs=num_freqs,
            value_dim=value_dim,
            omegas=omegas,
        )

        Vhat_fo = reconstruct_with_general_basis(V, mean_v, Q_fo_raw)
        v_recon_nmse_vs_degree_fo[i] = nmse_between_matrices(Vhat_fo, V)

    return v_recon_nmse_vs_degree_rp, v_recon_nmse_vs_degree_fo


# ============================================================
# Filter-bank realization helpers
# ============================================================

def vec_to_matrix_entries(vec_values: np.ndarray, d: int) -> np.ndarray:
    return vec_values.reshape(d, d, order="F")


def fit_fir_from_frequency_response(
    H_w: np.ndarray,
    omegas: np.ndarray,
    fir_len: int,
) -> np.ndarray:
    """
    H_w: shape (num_freqs, d_out, d_in)
    Returns FIR taps h[l, d_out, d_in] such that
        H(e^{jw}) ~= sum_{l=0}^{fir_len-1} h[l] e^{-jwl}
    """
    num_freqs, d_out, d_in = H_w.shape
    E = np.exp(-1j * omegas[:, None] * np.arange(fir_len)[None, :])
    taps = np.zeros((fir_len, d_out, d_in), dtype=np.complex128)

    for r in range(d_out):
        for c in range(d_in):
            h_rc, *_ = np.linalg.lstsq(E, H_w[:, r, c], rcond=None)
            taps[:, r, c] = h_rc

    return taps


class FIRBank:
    def __init__(self, taps: np.ndarray):
        self.taps = taps.astype(np.complex128)
        self.num_filters, self.fir_len, self.d_out, self.d_in = taps.shape
        self.input_hist = np.zeros((self.num_filters, self.fir_len, self.d_in), dtype=np.complex128)

    def reset(self):
        self.input_hist.fill(0.0)

    def step(self, u_t: np.ndarray) -> np.ndarray:
        self.input_hist[:, 1:, :] = self.input_hist[:, :-1, :]
        self.input_hist[:, 0, :] = u_t[None, :]
        outputs = np.zeros((self.num_filters, self.d_out), dtype=np.complex128)
        for j in range(self.num_filters):
            for l in range(self.fir_len):
                outputs[j] += self.taps[j, l] @ self.input_hist[j, l]
        return outputs


class SharedDenominatorIIRBank:
    def __init__(self, a_den: np.ndarray, b_num: np.ndarray, d_out: int, d_in: int):
        self.a_den = a_den.astype(np.complex128)
        self.b_num = b_num.astype(np.complex128)
        self.num_filters = b_num.shape[0]
        self.degree = b_num.shape[2]
        self.d_out = d_out
        self.d_in = d_in
        self.value_dim = d_out * d_in
        self.output_hist = np.zeros((self.num_filters, self.degree, d_out), dtype=np.complex128)
        self.input_hist = np.zeros((self.num_filters, self.degree, d_in), dtype=np.complex128)

    def reset(self):
        self.output_hist.fill(0.0)
        self.input_hist.fill(0.0)

    def step(self, u_t: np.ndarray) -> np.ndarray:
        self.input_hist[:, 1:, :] = self.input_hist[:, :-1, :]
        self.input_hist[:, 0, :] = u_t[None, :]
        outputs = np.zeros((self.num_filters, self.d_out), dtype=np.complex128)

        for j in range(self.num_filters):
            y = np.zeros((self.d_out,), dtype=np.complex128)
            for k in range(self.degree):
                y += -self.a_den[j, k] * self.output_hist[j, k]
                Bk = self.b_num[j, :, k].reshape(self.d_out, self.d_in, order="F")
                y += Bk @ self.input_hist[j, k]
            outputs[j] = y

        self.output_hist[:, 1:, :] = self.output_hist[:, :-1, :]
        self.output_hist[:, 0, :] = outputs
        return outputs


class FirstOrderBasisBank:
    def __init__(self, poles: np.ndarray, residues: np.ndarray, d_out: int, d_in: int):
        self.poles = poles.astype(np.complex128)
        self.residues = residues.astype(np.complex128)
        self.num_basis = residues.shape[0]
        self.num_terms = residues.shape[2]
        self.d_out = d_out
        self.d_in = d_in
        self.value_dim = d_out * d_in

        # Atomic first-order states: one state/output per (basis index, first-order term).
        self.state = np.zeros((self.num_basis, self.num_terms, d_out), dtype=np.complex128)

    def reset(self):
        self.state.fill(0.0)

    def step(self, u_t: np.ndarray) -> np.ndarray:
        # Return the full atomic first-order outputs, not the sum across terms.
        atom_outputs = np.zeros((self.num_basis, self.num_terms, self.d_out), dtype=np.complex128)
        for j in range(self.num_basis):
            for k in range(self.num_terms):
                Cjk = self.residues[j, :, k].reshape(self.d_out, self.d_in, order="F")
                self.state[j, k] = self.poles[j, k] * self.state[j, k] + Cjk @ u_t
                atom_outputs[j, k] = self.state[j, k]
        return atom_outputs


class RandomFirstOrderBasisBank:
    def __init__(
        self,
        num_basis: int,
        num_terms: int,
        d_out: int,
        d_in: int,
        rng: np.random.Generator,
        max_radius: float = 0.98,
        input_scale: float = 0.25,
    ):
        self.num_basis = num_basis
        self.num_terms = num_terms
        self.d_out = d_out
        self.d_in = d_in

        pole_mags = rng.uniform(0.0, max_radius, size=(num_basis, num_terms))
        pole_phases = rng.uniform(-np.pi, np.pi, size=(num_basis, num_terms))
        self.poles = pole_mags * np.exp(1j * pole_phases)

        self.input_maps = input_scale * (
            rng.standard_normal((num_basis, num_terms, d_out, d_in))
            + 1j * rng.standard_normal((num_basis, num_terms, d_out, d_in))
        ) / np.sqrt(2.0 * max(d_in, 1))

        # Atomic first-order states for the random baseline as well.
        self.state = np.zeros((num_basis, num_terms, d_out), dtype=np.complex128)

    def reset(self):
        self.state.fill(0.0)

    def step(self, u_t: np.ndarray) -> np.ndarray:
        atom_outputs = np.zeros((self.num_basis, self.num_terms, self.d_out), dtype=np.complex128)
        for j in range(self.num_basis):
            for k in range(self.num_terms):
                Cjk = self.input_maps[j, k]
                self.state[j, k] = self.poles[j, k] * self.state[j, k] + Cjk @ u_t
                atom_outputs[j, k] = self.state[j, k]
        return atom_outputs


# ============================================================
# Prediction helpers
# ============================================================

def build_fo_bank_from_fit_info(fit_info: list, num_freqs: int, d: int):
    _, _, fo_info = decompose_first_m_rp_columns_to_first_order(
        fit_info=fit_info,
        num_freqs=num_freqs,
        value_dim=d * d,
        omegas=np.linspace(0.0, np.pi, num_freqs, endpoint=True),
    )
    poles = np.stack([fo["poles"] for fo in fo_info], axis=0)
    residues = np.stack([fo["residues"] for fo in fo_info], axis=0)
    return FirstOrderBasisBank(poles=poles, residues=residues, d_out=d, d_in=d), fo_info


def build_random_fo_bank(num_basis: int, num_terms: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    return RandomFirstOrderBasisBank(
        num_basis=num_basis,
        num_terms=num_terms,
        d_out=d,
        d_in=d,
        rng=rng,
        max_radius=0.98,
        input_scale=0.25,
    )


def run_bank_and_collect_features(bank, y_seq: np.ndarray) -> np.ndarray:
    bank.reset()
    outputs = []
    for t in range(y_seq.shape[0]):
        out = bank.step(y_seq[t].astype(np.complex128))
        outputs.append(out.reshape(-1))
    return np.stack(outputs, axis=0)


def complex_linear_readout_train_and_predict(
    feature_seq: np.ndarray,
    target_seq: np.ndarray,
    reg: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z_train = feature_seq[:-1]
    Y_train = target_seq[1:-1]
    z_test = feature_seq[-1]
    y_true = target_seq[-1]

    G = Z_train.conj().T @ Z_train + reg * np.eye(Z_train.shape[1], dtype=np.complex128)
    I = np.eye(G.shape[0], dtype=G.dtype)
    rhs = Z_train.conj().T @ Y_train

    try:
        W = np.linalg.solve(G + reg * I, rhs)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(G + reg * I) @ rhs
    y_pred = z_test @ W
    return W, y_pred, y_true


def evaluate_prediction_nmse_over_chunks(
    x_all: np.ndarray,
    y_all: np.ndarray,
    history_len: int,
    bank_builder,
    target_kind: str = "x",
    reg: float = 1e-8,
) -> float:
    chunk_len = history_len + 1
    num_chunks = x_all.shape[0] // chunk_len
    preds = []
    trues = []

    for k in range(num_chunks):
        start = k * chunk_len
        end = (k + 1) * chunk_len
        y_chunk = y_all[start:end]
        x_chunk = x_all[start:end]

        bank = bank_builder()
        feat_seq = run_bank_and_collect_features(bank, y_chunk[:history_len])

        target_seq = x_chunk if target_kind == "x" else y_chunk
        _, pred, true = complex_linear_readout_train_and_predict(feat_seq, target_seq, reg=reg)
        preds.append(pred)
        trues.append(true)

    P = np.stack(preds, axis=0)
    T = np.stack(trues, axis=0)
    num = np.linalg.norm(P - T) ** 2
    den = np.linalg.norm(T) ** 2
    return float(np.real(num / max(den, 1e-15)))


def evaluate_steady_state_kalman_baseline_nmse_over_chunks(
    x_all: np.ndarray,
    y_all: np.ndarray,
    history_len: int,
    F_true: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    use_estimated_F: bool,
    max_em_iters: int = 50,
    em_tol: float = 1e-6,
    target_kind: str = "x",
) -> float:
    chunk_len = history_len + 1
    num_chunks = x_all.shape[0] // chunk_len
    preds = []
    trues = []

    for k in range(num_chunks):
        start = k * chunk_len
        end = (k + 1) * chunk_len
        y_chunk = y_all[start:end]
        x_chunk = x_all[start:end]
        y_hist = y_chunk[:history_len]

        F_used = select_chunk_F(
            y_hist=y_hist,
            F_true=F_true,
            Q=Q,
            R=R,
            use_estimated_F=use_estimated_F,
            max_em_iters=max_em_iters,
            em_tol=em_tol,
        )
        _, K_ss = solve_discrete_riccati_steady_state(F_used, Q, R)
        xhat_pred = run_steady_state_predictor(y_hist, F_used, K_ss)
        pred = xhat_pred[-1]

        true = x_chunk[-1] if target_kind == "x" else y_chunk[-1]
        preds.append(pred)
        trues.append(true)

    P = np.stack(preds, axis=0)
    T = np.stack(trues, axis=0)
    num = np.linalg.norm(P - T) ** 2
    den = np.linalg.norm(T) ** 2
    return float(np.real(num / max(den, 1e-15)))


def evaluate_full_kalman_baseline_nmse_over_chunks(
    x_all: np.ndarray,
    y_all: np.ndarray,
    history_len: int,
    F_true: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    use_estimated_F: bool,
    max_em_iters: int = 50,
    em_tol: float = 1e-6,
    target_kind: str = "x",
) -> float:
    chunk_len = history_len + 1
    num_chunks = x_all.shape[0] // chunk_len
    preds = []
    trues = []

    for k in range(num_chunks):
        start = k * chunk_len
        end = (k + 1) * chunk_len
        y_chunk = y_all[start:end]
        x_chunk = x_all[start:end]
        y_hist = y_chunk[:history_len]

        F_used = select_chunk_F(
            y_hist=y_hist,
            F_true=F_true,
            Q=Q,
            R=R,
            use_estimated_F=use_estimated_F,
            max_em_iters=max_em_iters,
            em_tol=em_tol,
        )
        xhat_pred = run_full_kalman_one_step_predictor(y_hist, F_used, Q, R)
        pred = xhat_pred[-1]

        true = x_chunk[-1] if target_kind == "x" else y_chunk[-1]
        preds.append(pred)
        trues.append(true)

    P = np.stack(preds, axis=0)
    T = np.stack(trues, axis=0)
    num = np.linalg.norm(P - T) ** 2
    den = np.linalg.norm(T) ** 2
    return float(np.real(num / max(den, 1e-15)))


# ============================================================
# Main
# ============================================================

def main():
    # ----------------------------
    # User-configurable parameters
    # ----------------------------
    F_template = np.array([
        [0.8, 0.6],
        [-0.4, 0.7],
    ], dtype=float)

    rho_mobility = 0.90
    target_power = 0.0025

    # NEW SETTING:
    #   False -> use perfect F_true everywhere
    #   True  -> use chunkwise estimated F everywhere
    use_estimated_F = True

    F_true, Q_process, C_target = build_unit_power_model_from_template(
        F_template=F_template,
        rho_mobility=rho_mobility,
        target_power=target_power,
    )
    R_obs = np.array([
        [1e-4, 0.0],
        [0.0, 1e-4],
    ], dtype=float)

    P_stat = solve_stationary_state_covariance(F_true, Q_process)
    print(
        f"Using unit-power channel model with rho={rho_mobility:.3f}. "
        f"target_power={target_power:.4e}, "
        f"trace(C_target)={np.real(np.trace(C_target)):.4e}, "
        f"trace(P_stat)={np.real(np.trace(P_stat)):.4e}"
    )
    print(f"use_estimated_F = {use_estimated_F}")

    history_len = 50
    num_chunks = 100
    num_test_chunks = 200
    num_freqs = 64

    max_em_iters = 50
    em_tol = 1e-6
    seed = 12345
    test_seed = 67890

    # Transfer-space basis settings
    rp_degree_for_m_curve = 4
    max_m = 5
    degree_vals = np.arange(1, 11)
    m_eval_for_degree_plot = max_m

    # Prediction settings
    target_kind = "y"
    readout_reg = 1e-8

    # -----------------------------------
    # Training chunks for transfer-space statistics / basis fixing
    # -----------------------------------
    results = generate_chunkwise_transfer_vectors(
        F_true=F_true,
        Q=Q_process,
        R=R_obs,
        history_len=history_len,
        num_chunks=num_chunks,
        num_freqs=num_freqs,
        use_estimated_F=use_estimated_F,
        seed=seed,
        max_em_iters=max_em_iters,
        em_tol=em_tol,
    )

    V = results["V"]
    print("Collected transfer-function sample matrix V with shape:", V.shape)

    mean_v, Vc, K_vv = estimate_empirical_complex_covariance(V)
    print("Empirical covariance K_vv shape:", K_vv.shape)

    evals, Q_eig = pca_from_covariance(K_vv)
    print("Top 10 eigenvalues:")
    print(np.real(evals[:10]))

    d = F_true.shape[0]
    value_dim = d * d

    # -----------------------------------
    # Transfer-space NMSE curves
    # -----------------------------------
    max_m = min(max_m, Q_eig.shape[1])
    m_vals, nmse_exact_q = compute_reconstruction_nmse_curve(
        V=V,
        mean_v=mean_v,
        Q=Q_eig,
        max_m=max_m,
    )

    (
        m_vals_rp,
        nmse_rp_q,
        nmse_fo_q,
        avg_q_fit_nmse_vs_m,
        avg_fo_fit_nmse_vs_m,
    ) = compute_rp_and_fo_basis_reconstruction_nmse_curve(
        V=V,
        mean_v=mean_v,
        Q_eig=Q_eig,
        num_freqs=num_freqs,
        value_dim=value_dim,
        degree=rp_degree_for_m_curve,
        max_m=max_m,
    )

    (
        v_recon_nmse_vs_degree_rp,
        v_recon_nmse_vs_degree_fo,
    ) = compute_degree_sweep_with_fo(
        V=V,
        mean_v=mean_v,
        Q_eig=Q_eig,
        num_freqs=num_freqs,
        value_dim=value_dim,
        m_eval=m_eval_for_degree_plot,
        degree_vals=degree_vals,
    )

    # -----------------------------------
    # Previously unseen testing chunks for prediction NMSE
    # -----------------------------------
    rng_test = np.random.default_rng(test_seed)
    total_test_T = num_test_chunks * (history_len + 1)
    x_test_all, y_test_all = generate_state_space_data(total_test_T, F_true, Q_process, R_obs, rng_test)

    pred_nmse_fo = np.zeros_like(m_vals, dtype=float)
    pred_nmse_random = np.zeros_like(m_vals, dtype=float)

    ss_kalman_baseline_nmse = evaluate_steady_state_kalman_baseline_nmse_over_chunks(
        x_all=x_test_all,
        y_all=y_test_all,
        history_len=history_len,
        F_true=F_true,
        Q=Q_process,
        R=R_obs,
        use_estimated_F=use_estimated_F,
        max_em_iters=max_em_iters,
        em_tol=em_tol,
        target_kind=target_kind,
    )
    print(f"Steady-state Kalman baseline NMSE: {ss_kalman_baseline_nmse:.4e}")

    full_kalman_baseline_nmse = evaluate_full_kalman_baseline_nmse_over_chunks(
        x_all=x_test_all,
        y_all=y_test_all,
        history_len=history_len,
        F_true=F_true,
        Q=Q_process,
        R=R_obs,
        use_estimated_F=use_estimated_F,
        max_em_iters=max_em_iters,
        em_tol=em_tol,
        target_kind=target_kind,
    )
    print(f"Full Kalman baseline NMSE: {full_kalman_baseline_nmse:.4e}")

    for i, m in enumerate(m_vals):
        if m == 0:
            pred_nmse_fo[i] = np.nan
            pred_nmse_random[i] = np.nan
            continue

        _, _, fit_info_m = fit_first_m_q_columns_with_rational_models(
            Q_eig=Q_eig,
            num_freqs=num_freqs,
            value_dim=value_dim,
            m=m,
            degree=rp_degree_for_m_curve,
            omegas=np.linspace(0.0, np.pi, num_freqs, endpoint=True),
        )

        def fo_builder_local(fit_info_local=fit_info_m):
            bank, _ = build_fo_bank_from_fit_info(
                fit_info=fit_info_local,
                num_freqs=num_freqs,
                d=d,
            )
            return bank

        pred_nmse_fo[i] = evaluate_prediction_nmse_over_chunks(
            x_all=x_test_all,
            y_all=y_test_all,
            history_len=history_len,
            bank_builder=fo_builder_local,
            target_kind=target_kind,
            reg=readout_reg,
        )

        random_seed_m = 1000 + int(m)

        def random_builder_local(m_local=m, seed_local=random_seed_m):
            return build_random_fo_bank(
                num_basis=m_local,
                num_terms=rp_degree_for_m_curve,
                d=d,
                seed=seed_local,
            )

        pred_nmse_random[i] = evaluate_prediction_nmse_over_chunks(
            x_all=x_test_all,
            y_all=y_test_all,
            history_len=history_len,
            bank_builder=random_builder_local,
            target_kind=target_kind,
            reg=readout_reg,
        )

        print(
            f"m={m:2d} | transfer NMSE exact={nmse_exact_q[i]:.4e}, "
            f"RP={nmse_rp_q[i]:.4e}, FO={nmse_fo_q[i]:.4e} | "
            f"prediction NMSE FO={pred_nmse_fo[i]:.4e}, "
            f"random={pred_nmse_random[i]:.4e}"
        )

    # -----------------------------------
    # Save arrays
    # -----------------------------------
    np.savez(
        "kalman_transfer_pca_rp_fo_prediction_stats.npz",
        F_template=F_template,
        rho_mobility=rho_mobility,
        target_power=target_power,
        use_estimated_F=use_estimated_F,
        C_target=C_target,
        F_true=F_true,
        Q_process=Q_process,
        R_obs=R_obs,
        history_len=history_len,
        num_chunks=num_chunks,
        num_test_chunks=num_test_chunks,
        num_freqs=num_freqs,
        V=V,
        mean_v=mean_v,
        K_vv=K_vv,
        evals=evals,
        Q_eig=Q_eig,
        F_hats=results["F_hats"],
        A_ps=results["A_ps"],
        B_ps=results["B_ps"],
        m_vals=m_vals,
        nmse_exact_q=nmse_exact_q,
        nmse_rp_q=nmse_rp_q,
        nmse_fo_q=nmse_fo_q,
        avg_q_fit_nmse_vs_m=avg_q_fit_nmse_vs_m,
        avg_fo_fit_nmse_vs_m=avg_fo_fit_nmse_vs_m,
        rp_degree_for_m_curve=rp_degree_for_m_curve,
        degree_vals=degree_vals,
        v_recon_nmse_vs_degree_rp=v_recon_nmse_vs_degree_rp,
        v_recon_nmse_vs_degree_fo=v_recon_nmse_vs_degree_fo,
        m_eval_for_degree_plot=m_eval_for_degree_plot,
        pred_nmse_fo=pred_nmse_fo,
        pred_nmse_random=pred_nmse_random,
        ss_kalman_baseline_nmse=ss_kalman_baseline_nmse,
        full_kalman_baseline_nmse=full_kalman_baseline_nmse,
        target_kind=target_kind,
    )

    # -----------------------------------
    # Plot 1: Transfer-space NMSE vs m
    # -----------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(m_vals, nmse_exact_q, marker="o", label="Exact PCA basis")
    plt.plot(
        m_vals_rp,
        nmse_rp_q,
        marker="s",
        label=f"RP-approximated PCA basis (degree={rp_degree_for_m_curve})",
    )
    plt.plot(
        m_vals_rp,
        nmse_fo_q,
        marker="^",
        label=f"First-order decomposed basis (degree={rp_degree_for_m_curve})",
    )
    plt.xlabel("Number of retained basis vectors (m)")
    plt.ylabel("Reconstruction NMSE of transfer samples")
    plt.title("Exact-Q vs RP-Q vs first-order decomposition")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("nmse_vs_m_exactQ_vs_rpQ_vs_fo.png", dpi=200)
    plt.show()

    # -----------------------------------
    # Plot 2: Transfer-space NMSE vs degree
    # -----------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(
        degree_vals,
        v_recon_nmse_vs_degree_rp,
        marker="s",
        label=f"V reconstruction NMSE using first {m_eval_for_degree_plot} RP columns",
    )
    plt.plot(
        degree_vals,
        v_recon_nmse_vs_degree_fo,
        marker="^",
        label=f"V reconstruction NMSE using first {m_eval_for_degree_plot} FO columns",
    )
    plt.xlabel("Rational polynomial degree")
    plt.ylabel("NMSE")
    plt.title("Effect of rational polynomial degree")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("nmse_vs_rational_degree.png", dpi=200)
    plt.show()

    # -----------------------------------
    # Plot 3: Prediction NMSE vs m on unseen testing chunks
    # -----------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(
        m_vals[1:],
        pred_nmse_fo[1:],
        marker="^",
        label=f"Configured filterbank + LS readout (degree={rp_degree_for_m_curve})",
    )
    plt.plot(
        m_vals[1:],
        pred_nmse_random[1:],
        marker="o",
        label=f"Random filterbank + LS readout ({rp_degree_for_m_curve} terms/basis)",
    )
    plt.axhline(
        ss_kalman_baseline_nmse,
        color="k",
        linestyle="--",
        linewidth=1.5,
        label="Steady-state Kalman baseline",
    )
    plt.axhline(
        full_kalman_baseline_nmse,
        color="tab:red",
        linestyle=":",
        linewidth=1.5,
        label="Full Kalman baseline",
    )
    plt.xlabel("Number of retained basis vectors (m)")
    plt.ylabel(f"Testing prediction NMSE vs true {target_kind}_{{N}}")
    plt.title("Configured FO vs random FO prediction NMSE on unseen chunks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.ylim((0, 1))
    plt.savefig("prediction_nmse_vs_m_fo_vs_random.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
