import numpy as np
import matplotlib.pyplot as plt


def symmetrize(M):
    return 0.5 * (M + M.T)


def project_to_stable_matrix(F, max_radius=0.99):
    """
    If spectral radius >= max_radius, scale F down so it is stable.
    """
    eigvals = np.linalg.eigvals(F)
    rho = np.max(np.abs(eigvals))
    if rho >= max_radius:
        F = F * (max_radius / (rho + 1e-15))
    return F


def solve_discrete_riccati_steady_state(F, Q, R, max_iter=10000, tol=1e-12):
    d = F.shape[0]
    P_minus = np.eye(d)

    for _ in range(max_iter):
        S = P_minus + R
        K = P_minus @ np.linalg.inv(S)
        P_plus = P_minus - K @ P_minus
        P_minus_next = F @ P_plus @ F.T + Q

        err = np.linalg.norm(P_minus_next - P_minus, ord='fro')
        P_minus = P_minus_next
        if err < tol:
            break

    S = P_minus + R
    K = P_minus @ np.linalg.inv(S)
    return P_minus, K


def generate_state_space_data(T, F, Q, R, rng):
    d = F.shape[0]
    x = np.zeros((T, d))
    y = np.zeros((T, d))

    x_prev = rng.multivariate_normal(np.zeros(d), np.eye(d))

    for t in range(T):
        w_t = rng.multivariate_normal(np.zeros(d), Q)
        x_t = F @ x_prev + w_t

        n_t = rng.multivariate_normal(np.zeros(d), R)
        y_t = x_t + n_t

        x[t] = x_t
        y[t] = y_t
        x_prev = x_t

    return x, y


def run_steady_state_predictor(y_hist, F, K):
    """
    Uses y[0],...,y[T_hist-1] to produce one-step-ahead predictions:
        xhat_pred[t] = xhat_{t+1|t}
    """
    T_hist, d = y_hist.shape
    A_p = F - F @ K
    B_p = F @ K

    xhat_pred = np.zeros((T_hist, d))
    s_prev = np.zeros(d)

    for t in range(T_hist):
        s_t = A_p @ s_prev + B_p @ y_hist[t]
        xhat_pred[t] = s_t
        s_prev = s_t

    return xhat_pred


def run_full_kalman_one_step_predictor(y_hist, F, Q, R):
    """
    Uses measurements y_hist = [y_0, ..., y_{T_hist-1}] to produce
    one-step-ahead predictions:
        xhat_pred[t] = xhat_{t+1|t}
    So if y_hist = y[:-1], then xhat_pred[-1] estimates x[-1].
    """
    T_hist, d = y_hist.shape

    xhat_plus = np.zeros(d)
    P_plus = np.eye(d)

    xhat_pred = np.zeros((T_hist, d))

    for t in range(T_hist):
        # Predict current state from previous posterior
        xhat_minus = F @ xhat_plus
        P_minus = F @ P_plus @ F.T + Q

        # Update with y_t
        S = P_minus + R
        K_t = P_minus @ np.linalg.inv(S)
        innovation = y_hist[t] - xhat_minus
        xhat_plus = xhat_minus + K_t @ innovation
        P_plus = (np.eye(d) - K_t) @ P_minus

        # One-step prediction for next state: x_{t+1|t}
        xhat_pred[t] = F @ xhat_plus

    return xhat_pred


def kalman_filter_identity_H(y, F, Q, R):
    """
    Kalman filter for:
        x_t = F x_{t-1} + w_t
        y_t = x_t + n_t
    with H = I.

    Returns filtered and predicted moments needed for RTS smoothing and EM.
    """
    T, d = y.shape

    x_pred = np.zeros((T, d))
    P_pred = np.zeros((T, d, d))
    x_filt = np.zeros((T, d))
    P_filt = np.zeros((T, d, d))
    K_hist = np.zeros((T, d, d))

    x_plus_prev = np.zeros(d)
    P_plus_prev = np.eye(d)

    for t in range(T):
        # Prediction
        x_minus = F @ x_plus_prev
        P_minus = F @ P_plus_prev @ F.T + Q
        P_minus = symmetrize(P_minus)

        # Update
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


def rts_smoother_with_lag_cov(y, F, Q, R):
    """
    Runs Kalman filter + RTS smoother, and computes lag-one smoothed covariances
    needed in the EM update for F.

    Returns:
        x_smooth[t]      = E[x_t | y_0,...,y_{T-1}]
        P_smooth[t]      = Cov[x_t | y_0,...,y_{T-1}]
        P_lag[t]         = Cov[x_t, x_{t-1} | y_0,...,y_{T-1}], for t>=1
    """
    x_pred, P_pred, x_filt, P_filt, K_hist = kalman_filter_identity_H(y, F, Q, R)

    T, d = y.shape

    x_smooth = np.zeros((T, d))
    P_smooth = np.zeros((T, d, d))
    J = np.zeros((max(T - 1, 1), d, d))

    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    # RTS backward pass
    for t in range(T - 2, -1, -1):
        J_t = P_filt[t] @ F.T @ np.linalg.inv(P_pred[t + 1])
        J[t] = J_t

        x_smooth[t] = x_filt[t] + J_t @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + J_t @ (P_smooth[t + 1] - P_pred[t + 1]) @ J_t.T
        P_smooth[t] = symmetrize(P_smooth[t])

    # Lag-one smoothed covariance:
    # P_lag[t] = Cov(x_t, x_{t-1} | y_0,...,y_{T-1}), t >= 1
    P_lag = np.zeros((T, d, d))

    if T >= 2:
        # Base case for final pair
        P_lag[T - 1] = (np.eye(d) - K_hist[T - 1]) @ F @ P_filt[T - 2]

        # Backward recursion
        for t in range(T - 2, 0, -1):
            P_lag[t] = (
                P_filt[t] @ J[t - 1].T
                + J[t] @ (P_lag[t + 1] - F @ P_filt[t]) @ J[t - 1].T
            )
            P_lag[t] = symmetrize(P_lag[t])

    return x_smooth, P_smooth, P_lag


def estimate_F_init_from_covariances(y, R, reg=1e-8):
    """
    Simple covariance-based initializer:
        Gamma_y(1) = F Gamma_x(0)
        Gamma_y(0) = Gamma_x(0) + R
    so
        F ~= Gamma_y(1) [Gamma_y(0) - R]^{-1}
    """
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


def estimate_F_from_y_em(y, Q, R, F_init=None, max_em_iters=50, tol=1e-6, verbose=False):
    """
    EM estimation of F for:
        x_t = F x_{t-1} + w_t,   w_t ~ N(0,Q)
        y_t = x_t + n_t,         n_t ~ N(0,R)
    with known Q and R.

    Only F is updated.
    """
    T, d = y.shape

    if T < 2:
        raise ValueError("Need at least T>=2 samples to estimate F.")

    if F_init is None:
        F = estimate_F_init_from_covariances(y, R)
    else:
        F = F_init.copy()

    F = project_to_stable_matrix(F, max_radius=0.99)

    for it in range(max_em_iters):
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

        delta = np.linalg.norm(F - F_old, ord='fro')

        if delta < tol:
            break

    return F


def one_step_last_sample_mc_nmse_all(F_true, Q, R, T=10, num_mc=2000, seed=0, F_model=None):
    """
    Computes three NMSEs for the final time index x[T-1]:

    1) steady-state Kalman predictor
    2) previous observation baseline: use y[T-2] as predictor for x[T-1]
    3) full Kalman filter one-step predictor

    Data are generated using F_true.
    Predictors use F_model. If F_model is None, use F_true.
    """
    if F_model is None:
        F_model = F_true

    _, K_ss = solve_discrete_riccati_steady_state(F_model, Q, R)

    rng = np.random.default_rng(seed)

    total_error_power_ss = 0.0
    total_error_power_prev_obs = 0.0
    total_error_power_full_kf = 0.0
    total_target_power = 0.0

    mse_runs_ss = np.zeros(num_mc)
    mse_runs_prev_obs = np.zeros(num_mc)
    mse_runs_full_kf = np.zeros(num_mc)
    target_power_runs = np.zeros(num_mc)

    for m in range(num_mc):
        x, y = generate_state_space_data(T=T, F=F_true, Q=Q, R=R, rng=rng)

        # Target is x[T-1]
        x_last = x[-1, :]

        # 1) Steady-state Kalman predictor: use y[0]...y[T-2] to predict x[T-1]
        xhat_pred_ss = run_steady_state_predictor(y[:-1, :], F_model, K_ss)
        xhat_last_ss = xhat_pred_ss[-1, :]

        # 2) Previous observation baseline: directly use y[T-2] as predictor for x[T-1]
        xhat_last_prev_obs = y[-2, :]

        # 3) Full Kalman filter one-step predictor: use y[0]...y[T-2] to predict x[T-1]
        xhat_pred_full_kf = run_full_kalman_one_step_predictor(y[:-1, :], F_model, Q, R)
        xhat_last_full_kf = xhat_pred_full_kf[-1, :]

        err_power_ss = np.sum(np.abs(xhat_last_ss - x_last) ** 2)
        err_power_prev_obs = np.sum(np.abs(xhat_last_prev_obs - x_last) ** 2)
        err_power_full_kf = np.sum(np.abs(xhat_last_full_kf - x_last) ** 2)
        target_power = np.sum(np.abs(x_last) ** 2)

        total_error_power_ss += err_power_ss
        total_error_power_prev_obs += err_power_prev_obs
        total_error_power_full_kf += err_power_full_kf
        total_target_power += target_power

        mse_runs_ss[m] = err_power_ss
        mse_runs_prev_obs[m] = err_power_prev_obs
        mse_runs_full_kf[m] = err_power_full_kf
        target_power_runs[m] = target_power

    nmse_ss = total_error_power_ss / max(total_target_power, 1e-15)
    nmse_prev_obs = total_error_power_prev_obs / max(total_target_power, 1e-15)
    nmse_full_kf = total_error_power_full_kf / max(total_target_power, 1e-15)

    mean_mse_per_run_ss = np.mean(mse_runs_ss)
    mean_mse_per_run_prev_obs = np.mean(mse_runs_prev_obs)
    mean_mse_per_run_full_kf = np.mean(mse_runs_full_kf)
    mean_target_power_per_run = np.mean(target_power_runs)

    return {
        'nmse_ss': nmse_ss,
        'nmse_prev_obs': nmse_prev_obs,
        'nmse_full_kf': nmse_full_kf,
        'mean_mse_per_run_ss': mean_mse_per_run_ss,
        'mean_mse_per_run_prev_obs': mean_mse_per_run_prev_obs,
        'mean_mse_per_run_full_kf': mean_mse_per_run_full_kf,
        'mean_target_power_per_run': mean_target_power_per_run,
    }


def estimate_F_for_setting(F_true, Q, R, T_train, seed, max_em_iters=50, tol=1e-6, verbose=False):
    rng = np.random.default_rng(seed)
    _, y_train = generate_state_space_data(T=T_train, F=F_true, Q=Q, R=R, rng=rng)

    F_init = estimate_F_init_from_covariances(y_train, R)
    F_hat = estimate_F_from_y_em(
        y=y_train,
        Q=Q,
        R=R,
        F_init=F_init,
        max_em_iters=max_em_iters,
        tol=tol,
        verbose=verbose
    )
    return F_hat


def sweep_fixed_Q_vary_R(F_true, Q_fixed, R_vars, T=10, num_mc=2000, seed=100,
                         T_train=1000, em_max_iters=50, em_tol=1e-6):
    nmse_ss_perfect_vals = []
    nmse_ss_est_vals = []
    nmse_prev_obs_vals = []
    nmse_full_kf_perfect_vals = []
    nmse_full_kf_est_vals = []

    mse_ss_perfect_vals = []
    mse_ss_est_vals = []
    mse_prev_obs_vals = []
    mse_full_kf_perfect_vals = []
    mse_full_kf_est_vals = []

    xpow_vals = []
    snr_db_vals = []
    F_est_list = []

    for i, rvar in enumerate(R_vars):
        R = np.array([[rvar, 0.0],
                      [0.0, rvar]], dtype=float)

        # Estimate F once for this (Q,R) setting using a separate training sequence
        F_hat = estimate_F_for_setting(
            F_true=F_true,
            Q=Q_fixed,
            R=R,
            T_train=T_train,
            seed=10_000 + seed + i,
            max_em_iters=em_max_iters,
            tol=em_tol,
            verbose=False
        )
        F_est_list.append(F_hat)

        results_perfect = one_step_last_sample_mc_nmse_all(
            F_true=F_true, Q=Q_fixed, R=R, T=T, num_mc=num_mc, seed=seed + i, F_model=F_true
        )

        results_est = one_step_last_sample_mc_nmse_all(
            F_true=F_true, Q=Q_fixed, R=R, T=T, num_mc=num_mc, seed=seed + i, F_model=F_hat
        )

        # SNR = E[||x_t||^2] / E[||n_t||^2]
        # Here E[||x_t||^2] is estimated by mean_target_power_per_run,
        # and E[||n_t||^2] = trace(R) since y_t = x_t + n_t.
        signal_power = results_perfect['mean_target_power_per_run']
        noise_power = np.trace(R)
        snr_linear = signal_power / max(noise_power, 1e-15)
        snr_db = 10.0 * np.log10(max(snr_linear, 1e-15))

        print(
            f'[fixed Q, vary R] R_var={rvar:.4e}, '
            f'SNR={snr_db:.2f} dB '
            f'-> perfect ss={results_perfect["nmse_ss"]:.6f}, '
            f'est ss={results_est["nmse_ss"]:.6f}, '
            f'prev-obs={results_perfect["nmse_prev_obs"]:.6f}, '
            f'perfect full-KF={results_perfect["nmse_full_kf"]:.6f}, '
            f'est full-KF={results_est["nmse_full_kf"]:.6f}'
        )

        nmse_ss_perfect_vals.append(results_perfect['nmse_ss'])
        nmse_ss_est_vals.append(results_est['nmse_ss'])
        nmse_prev_obs_vals.append(results_perfect['nmse_prev_obs'])
        nmse_full_kf_perfect_vals.append(results_perfect['nmse_full_kf'])
        nmse_full_kf_est_vals.append(results_est['nmse_full_kf'])

        mse_ss_perfect_vals.append(results_perfect['mean_mse_per_run_ss'])
        mse_ss_est_vals.append(results_est['mean_mse_per_run_ss'])
        mse_prev_obs_vals.append(results_perfect['mean_mse_per_run_prev_obs'])
        mse_full_kf_perfect_vals.append(results_perfect['mean_mse_per_run_full_kf'])
        mse_full_kf_est_vals.append(results_est['mean_mse_per_run_full_kf'])

        xpow_vals.append(signal_power)
        snr_db_vals.append(snr_db)

    return (
        np.array(nmse_ss_perfect_vals),
        np.array(nmse_ss_est_vals),
        np.array(nmse_prev_obs_vals),
        np.array(nmse_full_kf_perfect_vals),
        np.array(nmse_full_kf_est_vals),
        np.array(mse_ss_perfect_vals),
        np.array(mse_ss_est_vals),
        np.array(mse_prev_obs_vals),
        np.array(mse_full_kf_perfect_vals),
        np.array(mse_full_kf_est_vals),
        np.array(xpow_vals),
        np.array(snr_db_vals),
        np.array(F_est_list),
    )


def sweep_fixed_R_vary_Q(F_true, R_fixed, Q_vars, T=10, num_mc=2000, seed=200,
                         T_train=1000, em_max_iters=50, em_tol=1e-6):
    nmse_ss_perfect_vals = []
    nmse_ss_est_vals = []
    nmse_prev_obs_vals = []
    nmse_full_kf_perfect_vals = []
    nmse_full_kf_est_vals = []

    mse_ss_perfect_vals = []
    mse_ss_est_vals = []
    mse_prev_obs_vals = []
    mse_full_kf_perfect_vals = []
    mse_full_kf_est_vals = []

    xpow_vals = []
    F_est_list = []

    for i, qvar in enumerate(Q_vars):
        Q = np.array([[qvar, 0.0],
                      [0.0, qvar]], dtype=float)

        # Estimate F once for this (Q,R) setting using a separate training sequence
        F_hat = estimate_F_for_setting(
            F_true=F_true,
            Q=Q,
            R=R_fixed,
            T_train=T_train,
            seed=20_000 + seed + i,
            max_em_iters=em_max_iters,
            tol=em_tol,
            verbose=False
        )
        F_est_list.append(F_hat)

        results_perfect = one_step_last_sample_mc_nmse_all(
            F_true=F_true, Q=Q, R=R_fixed, T=T, num_mc=num_mc, seed=seed + i, F_model=F_true
        )

        results_est = one_step_last_sample_mc_nmse_all(
            F_true=F_true, Q=Q, R=R_fixed, T=T, num_mc=num_mc, seed=seed + i, F_model=F_hat
        )

        print(
            f'[fixed R, vary Q] Q_var={qvar:.4e} '
            f'-> perfect ss={results_perfect["nmse_ss"]:.6f}, '
            f'est ss={results_est["nmse_ss"]:.6f}, '
            f'prev-obs={results_perfect["nmse_prev_obs"]:.6f}, '
            f'perfect full-KF={results_perfect["nmse_full_kf"]:.6f}, '
            f'est full-KF={results_est["nmse_full_kf"]:.6f}'
        )

        nmse_ss_perfect_vals.append(results_perfect['nmse_ss'])
        nmse_ss_est_vals.append(results_est['nmse_ss'])
        nmse_prev_obs_vals.append(results_perfect['nmse_prev_obs'])
        nmse_full_kf_perfect_vals.append(results_perfect['nmse_full_kf'])
        nmse_full_kf_est_vals.append(results_est['nmse_full_kf'])

        mse_ss_perfect_vals.append(results_perfect['mean_mse_per_run_ss'])
        mse_ss_est_vals.append(results_est['mean_mse_per_run_ss'])
        mse_prev_obs_vals.append(results_perfect['mean_mse_per_run_prev_obs'])
        mse_full_kf_perfect_vals.append(results_perfect['mean_mse_per_run_full_kf'])
        mse_full_kf_est_vals.append(results_est['mean_mse_per_run_full_kf'])

        xpow_vals.append(results_perfect['mean_target_power_per_run'])

    return (
        np.array(nmse_ss_perfect_vals),
        np.array(nmse_ss_est_vals),
        np.array(nmse_prev_obs_vals),
        np.array(nmse_full_kf_perfect_vals),
        np.array(nmse_full_kf_est_vals),
        np.array(mse_ss_perfect_vals),
        np.array(mse_ss_est_vals),
        np.array(mse_prev_obs_vals),
        np.array(mse_full_kf_perfect_vals),
        np.array(mse_full_kf_est_vals),
        np.array(xpow_vals),
        np.array(F_est_list),
    )


def main():
    F = np.array([
        [0.8, 0.6],
        [-0.4, 0.7],
    ], dtype=float)

    eigvals = np.linalg.eigvals(F)

    if np.max(np.abs(eigvals)) >= 1.0:
        raise ValueError('F must be stable.')

    # Balanced fixed values:
    # - for the R sweep, use a low diagonal Q
    # - for the Q sweep, use a low diagonal R
    Q_fixed_for_R_sweep = np.array([
        [0.0001, 0.0],
        [0.0, 0.0001],
    ], dtype=float)

    R_fixed_for_Q_sweep = np.array([
        [0.0001, 0.0],
        [0.0, 0.0001],
    ], dtype=float)

    T = 20
    num_mc = 100

    # Separate training-sequence length used to estimate F via EM
    T_train = 20
    em_max_iters = 50
    em_tol = 1e-6

    R_vars = np.logspace(-4, -0.5, 10)
    Q_vars = np.logspace(-4, -0.5, 10)

    print('\nUsing fixed Q for R sweep:')
    print(Q_fixed_for_R_sweep)

    print('\nUsing fixed R for Q sweep:')
    print(R_fixed_for_Q_sweep)

    (
        nmse_ss_vs_R_perfect,
        nmse_ss_vs_R_est,
        nmse_prev_obs_vs_R,
        nmse_full_kf_vs_R_perfect,
        nmse_full_kf_vs_R_est,
        mse_ss_vs_R_perfect,
        mse_ss_vs_R_est,
        mse_prev_obs_vs_R,
        mse_full_kf_vs_R_perfect,
        mse_full_kf_vs_R_est,
        xpow_vs_R,
        snr_db_vs_R,
        F_est_vs_R,
    ) = sweep_fixed_Q_vary_R(
        F_true=F,
        Q_fixed=Q_fixed_for_R_sweep,
        R_vars=R_vars,
        T=T,
        num_mc=num_mc,
        seed=100,
        T_train=T_train,
        em_max_iters=em_max_iters,
        em_tol=em_tol,
    )

    (
        nmse_ss_vs_Q_perfect,
        nmse_ss_vs_Q_est,
        nmse_prev_obs_vs_Q,
        nmse_full_kf_vs_Q_perfect,
        nmse_full_kf_vs_Q_est,
        mse_ss_vs_Q_perfect,
        mse_ss_vs_Q_est,
        mse_prev_obs_vs_Q,
        mse_full_kf_vs_Q_perfect,
        mse_full_kf_vs_Q_est,
        xpow_vs_Q,
        F_est_vs_Q,
    ) = sweep_fixed_R_vary_Q(
        F_true=F,
        R_fixed=R_fixed_for_Q_sweep,
        Q_vars=Q_vars,
        T=T,
        num_mc=num_mc,
        seed=200,
        T_train=T_train,
        em_max_iters=em_max_iters,
        em_tol=em_tol,
    )

    np.savez(
        'kalman_nmse_balanced_sweeps_results_with_em_F_estimation.npz',
        F_true=F,
        Q_fixed_for_R_sweep=Q_fixed_for_R_sweep,
        R_fixed_for_Q_sweep=R_fixed_for_Q_sweep,
        T=T,
        num_mc=num_mc,
        T_train=T_train,
        em_max_iters=em_max_iters,
        em_tol=em_tol,
        R_vars=R_vars,
        Q_vars=Q_vars,

        nmse_ss_vs_R_perfect=nmse_ss_vs_R_perfect,
        nmse_ss_vs_R_est=nmse_ss_vs_R_est,
        nmse_prev_obs_vs_R=nmse_prev_obs_vs_R,
        nmse_full_kf_vs_R_perfect=nmse_full_kf_vs_R_perfect,
        nmse_full_kf_vs_R_est=nmse_full_kf_vs_R_est,
        mse_ss_vs_R_perfect=mse_ss_vs_R_perfect,
        mse_ss_vs_R_est=mse_ss_vs_R_est,
        mse_prev_obs_vs_R=mse_prev_obs_vs_R,
        mse_full_kf_vs_R_perfect=mse_full_kf_vs_R_perfect,
        mse_full_kf_vs_R_est=mse_full_kf_vs_R_est,
        xpow_vs_R=xpow_vs_R,
        snr_db_vs_R=snr_db_vs_R,
        F_est_vs_R=F_est_vs_R,

        nmse_ss_vs_Q_perfect=nmse_ss_vs_Q_perfect,
        nmse_ss_vs_Q_est=nmse_ss_vs_Q_est,
        nmse_prev_obs_vs_Q=nmse_prev_obs_vs_Q,
        nmse_full_kf_vs_Q_perfect=nmse_full_kf_vs_Q_perfect,
        nmse_full_kf_vs_Q_est=nmse_full_kf_vs_Q_est,
        mse_ss_vs_Q_perfect=mse_ss_vs_Q_perfect,
        mse_ss_vs_Q_est=mse_ss_vs_Q_est,
        mse_prev_obs_vs_Q=mse_prev_obs_vs_Q,
        mse_full_kf_vs_Q_perfect=mse_full_kf_vs_Q_perfect,
        mse_full_kf_vs_Q_est=mse_full_kf_vs_Q_est,
        xpow_vs_Q=xpow_vs_Q,
        F_est_vs_Q=F_est_vs_Q
    )

    # Plot: fixed Q, vary R
    plt.figure(figsize=(8, 5.5))
    plt.semilogx(R_vars, nmse_ss_vs_R_perfect, marker='o', label='Steady-state KF (perfect F)')
    plt.semilogx(R_vars, nmse_ss_vs_R_est, marker='o', linestyle='--', label='Steady-state KF (estimated F)')
    plt.semilogx(R_vars, nmse_prev_obs_vs_R, marker='s', label='Previous observation baseline')
    plt.semilogx(R_vars, nmse_full_kf_vs_R_perfect, marker='^', label='Full KF (perfect F)')
    plt.semilogx(R_vars, nmse_full_kf_vs_R_est, marker='^', linestyle='--', label='Full KF (estimated F)')
    plt.xlabel('Observation-noise variance scale')
    plt.ylabel('One-step last-sample NMSE')
    plt.title(f'Fixed Q, vary R   (EM-based F estimation, T_train={T_train})')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.ylim(0.0, 1.0)
    plt.savefig('nmse_vs_R_with_em_estimated_F.png', dpi=200)

    # Plot: fixed R, vary Q
    plt.figure(figsize=(8, 5.5))
    plt.semilogx(Q_vars, nmse_ss_vs_Q_perfect, marker='o', label='Steady-state KF (perfect F)')
    plt.semilogx(Q_vars, nmse_ss_vs_Q_est, marker='o', linestyle='--', label='Steady-state KF (estimated F)')
    plt.semilogx(Q_vars, nmse_prev_obs_vs_Q, marker='s', label='Previous observation baseline')
    plt.semilogx(Q_vars, nmse_full_kf_vs_Q_perfect, marker='^', label='Full KF (perfect F)')
    plt.semilogx(Q_vars, nmse_full_kf_vs_Q_est, marker='^', linestyle='--', label='Full KF (estimated F)')
    plt.xlabel('Process-noise variance scale')
    plt.ylabel('One-step last-sample NMSE')
    plt.title(f'Fixed R, vary Q   (EM-based F estimation, T_train={T_train})')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.ylim(0.0, 1.0)
    plt.savefig('nmse_vs_Q_with_em_estimated_F.png', dpi=200)

    plt.show()


if __name__ == '__main__':
    main()