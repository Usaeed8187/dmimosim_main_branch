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


def one_step_last_sample_mc_nmse(F, Q, R, T=10, num_mc=2000, seed=0):
    _, K = solve_discrete_riccati_steady_state(F, Q, R)

    rng = np.random.default_rng(seed)

    total_error_power = 0.0
    total_target_power = 0.0
    mse_runs = np.zeros(num_mc)
    target_power_runs = np.zeros(num_mc)

    for m in range(num_mc):
        x, y = generate_state_space_data(T=T, F=F, Q=Q, R=R, rng=rng)

        xhat_pred = run_steady_state_predictor(y[:-1, :], F, K)
        xhat_last = xhat_pred[-1, :]
        x_last = x[-1, :]

        err_power = np.sum(np.abs(xhat_last - x_last) ** 2)
        target_power = np.sum(np.abs(x_last) ** 2)

        total_error_power += err_power
        total_target_power += target_power
        mse_runs[m] = err_power
        target_power_runs[m] = target_power

    nmse_stable = total_error_power / max(total_target_power, 1e-15)
    mean_mse_per_run = np.mean(mse_runs)
    mean_target_power_per_run = np.mean(target_power_runs)

    return nmse_stable, mean_mse_per_run, mean_target_power_per_run


def sweep_fixed_Q_vary_R(F, Q_fixed, R_vars, T=10, num_mc=2000, seed=100):
    nmse_vals = []
    mse_vals = []
    xpow_vals = []

    for i, rvar in enumerate(R_vars):
        R = np.array([[rvar, 0.0],
                      [0.0, rvar]], dtype=float)

        nmse, mse, xpow = one_step_last_sample_mc_nmse(
            F=F, Q=Q_fixed, R=R, T=T, num_mc=num_mc, seed=seed + i
        )

        print(f'[fixed Q, vary R] R_var={rvar:.4e} -> stable NMSE={nmse:.6f}')
        nmse_vals.append(nmse)
        mse_vals.append(mse)
        xpow_vals.append(xpow)

    return np.array(nmse_vals), np.array(mse_vals), np.array(xpow_vals)


def sweep_fixed_R_vary_Q(F, R_fixed, Q_vars, T=10, num_mc=2000, seed=200):
    nmse_vals = []
    mse_vals = []
    xpow_vals = []

    for i, qvar in enumerate(Q_vars):
        Q = np.array([[qvar, 0.0],
                      [0.0, qvar]], dtype=float)

        nmse, mse, xpow = one_step_last_sample_mc_nmse(
            F=F, Q=Q, R=R_fixed, T=T, num_mc=num_mc, seed=seed + i
        )

        print(f'[fixed R, vary Q] Q_var={qvar:.4e} -> stable NMSE={nmse:.6f}')
        nmse_vals.append(nmse)
        mse_vals.append(mse)
        xpow_vals.append(xpow)

    return np.array(nmse_vals), np.array(mse_vals), np.array(xpow_vals)


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
    # This avoids one covariance being much larger than the other across most
    # of the sweep range.
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

    nmse_vs_R, mse_vs_R, xpow_vs_R = sweep_fixed_Q_vary_R(
        F=F,
        Q_fixed=Q_fixed_for_R_sweep,
        R_vars=R_vars,
        T=T,
        num_mc=num_mc,
        seed=100
    )

    nmse_vs_Q, mse_vs_Q, xpow_vs_Q = sweep_fixed_R_vary_Q(
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
        nmse_vs_R=nmse_vs_R,
        mse_vs_R=mse_vs_R,
        xpow_vs_R=xpow_vs_R,
        nmse_vs_Q=nmse_vs_Q,
        mse_vs_Q=mse_vs_Q,
        xpow_vs_Q=xpow_vs_Q
    )

    plt.figure(figsize=(7, 5))
    plt.semilogx(R_vars, nmse_vs_R, marker='o')
    plt.xlabel('Observation-noise variance scale')
    plt.ylabel('Stable one-step last-sample NMSE')
    plt.title('Fixed Q, vary R')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('nmse_vs_R.png', dpi=200)

    plt.figure(figsize=(7, 5))
    plt.semilogx(Q_vars, nmse_vs_Q, marker='o')
    plt.xlabel('Process-noise variance scale')
    plt.ylabel('Stable one-step last-sample NMSE')
    plt.title('Fixed R, vary Q')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('nmse_vs_Q.png', dpi=200)

    plt.show()


if __name__ == '__main__':
    main()