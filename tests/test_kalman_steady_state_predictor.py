import numpy as np
import matplotlib.pyplot as plt


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


def one_step_last_sample_mc_nmse_all(F, Q, R, T=10, num_mc=2000, seed=0):
    """
    Computes three NMSEs for the final time index x[T-1]:

    1) steady-state Kalman predictor
    2) previous observation baseline: use y[T-2] to predict x[T-1]
    3) full Kalman filter one-step predictor
    """
    _, K_ss = solve_discrete_riccati_steady_state(F, Q, R)

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
        x, y = generate_state_space_data(T=T, F=F, Q=Q, R=R, rng=rng)

        # Target is x[T-1]
        x_last = x[-1, :]

        # 1) Steady-state Kalman predictor: use y[0]...y[T-2] to predict x[T-1]
        xhat_pred_ss = run_steady_state_predictor(y[:-1, :], F, K_ss)
        xhat_last_ss = xhat_pred_ss[-1, :]

        # 2) Previous observation baseline: directly use y[T-2] as predictor for x[T-1]
        xhat_last_prev_obs = y[-2, :]

        # 3) Full Kalman filter one-step predictor: use y[0]...y[T-2] to predict x[T-1]
        xhat_pred_full_kf = run_full_kalman_one_step_predictor(y[:-1, :], F, Q, R)
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


def sweep_fixed_Q_vary_R(F, Q_fixed, R_vars, T=10, num_mc=2000, seed=100):
    nmse_ss_vals = []
    nmse_prev_obs_vals = []
    nmse_full_kf_vals = []

    mse_ss_vals = []
    mse_prev_obs_vals = []
    mse_full_kf_vals = []

    xpow_vals = []

    for i, rvar in enumerate(R_vars):
        R = np.array([[rvar, 0.0],
                      [0.0, rvar]], dtype=float)

        results = one_step_last_sample_mc_nmse_all(
            F=F, Q=Q_fixed, R=R, T=T, num_mc=num_mc, seed=seed + i
        )

        print(
            f'[fixed Q, vary R] R_var={rvar:.4e} '
            f'-> steady-state NMSE={results["nmse_ss"]:.6f}, '
            f'prev-obs NMSE={results["nmse_prev_obs"]:.6f}, '
            f'full-KF NMSE={results["nmse_full_kf"]:.6f}'
        )

        nmse_ss_vals.append(results['nmse_ss'])
        nmse_prev_obs_vals.append(results['nmse_prev_obs'])
        nmse_full_kf_vals.append(results['nmse_full_kf'])

        mse_ss_vals.append(results['mean_mse_per_run_ss'])
        mse_prev_obs_vals.append(results['mean_mse_per_run_prev_obs'])
        mse_full_kf_vals.append(results['mean_mse_per_run_full_kf'])

        xpow_vals.append(results['mean_target_power_per_run'])

    return (
        np.array(nmse_ss_vals),
        np.array(nmse_prev_obs_vals),
        np.array(nmse_full_kf_vals),
        np.array(mse_ss_vals),
        np.array(mse_prev_obs_vals),
        np.array(mse_full_kf_vals),
        np.array(xpow_vals),
    )


def sweep_fixed_R_vary_Q(F, R_fixed, Q_vars, T=10, num_mc=2000, seed=200):
    nmse_ss_vals = []
    nmse_prev_obs_vals = []
    nmse_full_kf_vals = []

    mse_ss_vals = []
    mse_prev_obs_vals = []
    mse_full_kf_vals = []

    xpow_vals = []

    for i, qvar in enumerate(Q_vars):
        Q = np.array([[qvar, 0.0],
                      [0.0, qvar]], dtype=float)

        results = one_step_last_sample_mc_nmse_all(
            F=F, Q=Q, R=R_fixed, T=T, num_mc=num_mc, seed=seed + i
        )

        print(
            f'[fixed R, vary Q] Q_var={qvar:.4e} '
            f'-> steady-state NMSE={results["nmse_ss"]:.6f}, '
            f'prev-obs NMSE={results["nmse_prev_obs"]:.6f}, '
            f'full-KF NMSE={results["nmse_full_kf"]:.6f}'
        )

        nmse_ss_vals.append(results['nmse_ss'])
        nmse_prev_obs_vals.append(results['nmse_prev_obs'])
        nmse_full_kf_vals.append(results['nmse_full_kf'])

        mse_ss_vals.append(results['mean_mse_per_run_ss'])
        mse_prev_obs_vals.append(results['mean_mse_per_run_prev_obs'])
        mse_full_kf_vals.append(results['mean_mse_per_run_full_kf'])

        xpow_vals.append(results['mean_target_power_per_run'])

    return (
        np.array(nmse_ss_vals),
        np.array(nmse_prev_obs_vals),
        np.array(nmse_full_kf_vals),
        np.array(mse_ss_vals),
        np.array(mse_prev_obs_vals),
        np.array(mse_full_kf_vals),
        np.array(xpow_vals),
    )


def main():
    F = np.array([
        [0.8, 0.6],
        [-0.4, 0.7],
    ], dtype=float)

    eigvals = np.linalg.eigvals(F)
    print('Eigenvalues of F:', eigvals)
    print('Max |eig|:', np.max(np.abs(eigvals)))

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

    T = 8
    num_mc = 2000

    R_vars = np.logspace(-4, -0.5, 10)
    Q_vars = np.logspace(-4, -0.5, 10)

    print('\nUsing fixed Q for R sweep:')
    print(Q_fixed_for_R_sweep)

    print('\nUsing fixed R for Q sweep:')
    print(R_fixed_for_Q_sweep)

    (
        nmse_ss_vs_R,
        nmse_prev_obs_vs_R,
        nmse_full_kf_vs_R,
        mse_ss_vs_R,
        mse_prev_obs_vs_R,
        mse_full_kf_vs_R,
        xpow_vs_R,
    ) = sweep_fixed_Q_vary_R(
        F=F,
        Q_fixed=Q_fixed_for_R_sweep,
        R_vars=R_vars,
        T=T,
        num_mc=num_mc,
        seed=100
    )

    (
        nmse_ss_vs_Q,
        nmse_prev_obs_vs_Q,
        nmse_full_kf_vs_Q,
        mse_ss_vs_Q,
        mse_prev_obs_vs_Q,
        mse_full_kf_vs_Q,
        xpow_vs_Q,
    ) = sweep_fixed_R_vary_Q(
        F=F,
        R_fixed=R_fixed_for_Q_sweep,
        Q_vars=Q_vars,
        T=T,
        num_mc=num_mc,
        seed=200
    )

    np.savez(
        'kalman_nmse_balanced_sweeps_results.npz',
        F=F,
        Q_fixed_for_R_sweep=Q_fixed_for_R_sweep,
        R_fixed_for_Q_sweep=R_fixed_for_Q_sweep,
        T=T,
        num_mc=num_mc,
        R_vars=R_vars,
        Q_vars=Q_vars,

        nmse_ss_vs_R=nmse_ss_vs_R,
        nmse_prev_obs_vs_R=nmse_prev_obs_vs_R,
        nmse_full_kf_vs_R=nmse_full_kf_vs_R,
        mse_ss_vs_R=mse_ss_vs_R,
        mse_prev_obs_vs_R=mse_prev_obs_vs_R,
        mse_full_kf_vs_R=mse_full_kf_vs_R,
        xpow_vs_R=xpow_vs_R,

        nmse_ss_vs_Q=nmse_ss_vs_Q,
        nmse_prev_obs_vs_Q=nmse_prev_obs_vs_Q,
        nmse_full_kf_vs_Q=nmse_full_kf_vs_Q,
        mse_ss_vs_Q=mse_ss_vs_Q,
        mse_prev_obs_vs_Q=mse_prev_obs_vs_Q,
        mse_full_kf_vs_Q=mse_full_kf_vs_Q,
        xpow_vs_Q=xpow_vs_Q
    )

    plt.figure(figsize=(7, 5))
    plt.semilogx(R_vars, nmse_ss_vs_R, marker='o', label='Steady-state KF predictor')
    plt.semilogx(R_vars, nmse_prev_obs_vs_R, marker='s', label='Previous observation baseline')
    plt.semilogx(R_vars, nmse_full_kf_vs_R, marker='^', label='Full KF predictor')
    plt.xlabel('Observation-noise variance scale')
    plt.ylabel('One-step last-sample NMSE')
    plt.title('Fixed Q, vary R')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('nmse_vs_R.png', dpi=200)

    plt.figure(figsize=(7, 5))
    plt.semilogx(Q_vars, nmse_ss_vs_Q, marker='o', label='Steady-state KF predictor')
    plt.semilogx(Q_vars, nmse_prev_obs_vs_Q, marker='s', label='Previous observation baseline')
    plt.semilogx(Q_vars, nmse_full_kf_vs_Q, marker='^', label='Full KF predictor')
    plt.xlabel('Process-noise variance scale')
    plt.ylabel('One-step last-sample NMSE')
    plt.title('Fixed R, vary Q')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('nmse_vs_Q.png', dpi=200)

    plt.show()


if __name__ == '__main__':
    main()