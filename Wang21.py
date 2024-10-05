import numpy as np
import matplotlib.pyplot as plt
import common_functions as cf
import time
import warnings
import cvxpy as cp

# ----------- Main Functions ----------- #
def main(pow_tx: float, pow_noise: float, H: np.ndarray, T: np.ndarray, R: np.ndarray, theta_init: np.ndarray,
         theta_min: np.ndarray, theta_max: np.ndarray, beta_min: np.ndarray, psi: np.ndarray, mu: np.ndarray,
         main_loop: cf.loop_crit, ps_loop: cf.loop_crit, WF_loop: cf.loop_crit):
    assert pow_tx > 0, \
        f"Input pow_tx must be positive - It's in watt unit. If your input is still in dBm or dBw,convert it to watt first." \
        f"\nCurrent input pow_tx = {pow_tx}"
    assert pow_noise > 0, \
        f"Input pow_noise must be positive - It's in watt unit. If your input is still in dBm or dBw,convert it to watt first." \
        f"\nCurrent input pow_noise = {pow_noise}"
    assert H.ndim == 2 and T.ndim == 2 and R.ndim == 2 and theta_init.ndim == 2, \
        f"The dimension of channels H, T, R, and theta_init must be 2. \nCurrent H.ndim = {H.ndim}. \nCurrent T.ndim = {T.ndim}." \
        f"\nCurrent R.ndim = {R.ndim}. \nCurrent theta_init.ndim = {theta_init.ndim}"
    assert theta_min.ndim <= 1 and theta_min.size == 1, \
        f"Input theta_min must be a scalar with ndim <= 1.\nCurrent theta_min = {theta_min}."
    assert theta_max.ndim <= 1 and theta_max.size == 1, \
        f"Input theta_max must be a scalar with ndim <= 1.\nCurrent theta_max = {theta_max}."
    assert beta_min.ndim <= 1 and beta_min.size == 1 and np.all(beta_min >= 0), \
        f"Input beta_min must be a scalar with ndim <= 1 whose value is non-negative. \nCurrent beta_min = {beta_min}."
    assert psi.ndim <= 1 and psi.size == 1 and np.all(psi > -np.pi) and np.all(psi <= np.pi), \
        f"Input psi mus be a scalar with ndim <= 1 whose value is between -pi and pi. \nCurrent psi = {psi}"
    assert mu.ndim <= 1 and mu.size == 1 and np.all(mu >= 0), \
        f"Input alpha must be a scalar with ndim <= 1 whose value is non-negative. \nCurrent alpha = {mu}."
    assert np.all(theta_init > theta_min) and np.all(theta_init <= theta_max), \
        f"All values of theta_init_temp must be between-pi and pi. \nCurrent theta_init_temp = {theta_init}"

    Nr = H.shape[0]
    Nt = H.shape[1]
    M = T.shape[0]
    assert T.shape[1] == Nt,\
        f"The value of H.shape[1] and and T.shape[1] must be the same. \nCurrent H.shape[1] = {Nt}." \
        f"\nCurrent T.shape[1] = {T.shape[1]}."
    assert R.shape[0] == Nr,\
        f"The value of H.shape[0] and R.shape[0] must be the same. \nCurrent value of H.shape[0] = {Nr}." \
        f"\nCurrent value of R.shape[0] = {R.shape[0]}."
    assert R.shape[1] == M,\
        f"The value of T.shape[0] and R.shape[1] must be the same. \nCurrent value of T.shape[0] = {M}." \
        f"\nCurrent value of R.shape[1] = {R.shape[1]}."
    assert theta_init.shape[1] == M,\
        f"The value of theta_init.shape[1] must be {M} in order to consistent with the size of T and R." \
        f"\nCurrent theta_init.shape = {theta_init.shape}"
    L = theta_init.shape[0]  # ; print(f"L = {L}")

    time_start = time.time()

    # Choosing theta_init_used
    theta_init_used = 0
    rate_highest = 0
    # print(f"theta_init.shape = {theta_init.shape}")
    for l in range(L):
        # print(f"l = {l} -----------")
        theta_init_temp = theta_init[l, :]  # ; print(f"theta_init_temp = {theta_init_temp}")
        _, rate_temp = get_opt_Q_hat(H, T, R, pow_tx, pow_noise, theta_init_temp, theta_min, theta_max, beta_min, psi, mu, WF_loop)  # ; print(f"rate_temp = {rate_temp}")
        if rate_temp > rate_highest:
            rate_highest = rate_temp.copy()
            theta_init_used = theta_init_temp.copy()

    rate_diff = np.inf
    iter_th = 0
    theta = theta_init_used.copy()
    rate_trace = np.expand_dims(rate_highest, axis=0)  # ; print(f"rate_trace = {rate_trace}")
    rate = 0  # Dummy
    Q_hat = 0  # Which is defined as Q = Q_hat @ Q_hat.T.conj()
    while rate_diff > main_loop.tol_out and iter_th < main_loop.iterlim_out:
        iter_th += 1  # ; print(f"main_loop iter_th = {iter_th} ---------------------")

        # Optimization w.r.t. Precoder T
        Q_hat, rate = get_opt_Q_hat(H, T, R, pow_tx, pow_noise, theta, theta_min, theta_max, beta_min, psi, mu, WF_loop)
        # print(f"   rate_wang (opt Q_hat) = {rate}")

        # Optimization w.r.t. phase-shift theta
        theta, rate = get_opt_theta(H, T, R, Q_hat, pow_tx, pow_noise, theta, theta_min, theta_max, beta_min, psi, mu, ps_loop)
        # print(f"   rate_wang (opt theta) = {rate}")

        rate_trace = np.concatenate((rate_trace, np.expand_dims(rate, axis=0)), axis=0)
        rate_diff = np.abs(rate_trace[-1] - rate_trace[-2])

    # print(f"   rate_trace (main, wang) = {rate_trace}")

    if iter_th >= main_loop.iterlim_out:
        warnings.warn(f"WARNING! 'iter_th' reaches the limit (={iter_th}), the iteration may not converge")

    time_exec = time.time() - time_start
    return rate, Q_hat, theta, time_exec, rate_trace

# ----------- Precoder Optimization ----------- #
def get_beta(theta: np.ndarray, beta_min: np.ndarray, psi: np.ndarray, mu: np.ndarray):
    return (1 - beta_min) * ((np.sin(theta - psi) + 1) / 2)**mu + beta_min

def get_Phi(theta: np.ndarray, beta: np.ndarray):
    assert theta.ndim == 1, \
        f"Input theta must be a 1D-array with ndim=1.\nCurrent theta.ndim = {theta.ndim}."
    assert np.array_equal(theta.shape, beta.shape), \
        f"The shape of theta and beta must be the same." \
        f"\nCurrent input theta.shape = {theta.shape}. \nCurrent input beta.shape = {beta.shape}."
    return np.diag(beta * np.exp(1j * theta))

def get_H_hat(H, T, R, theta, beta_min, psi, mu):
    beta = get_beta(theta, beta_min, psi, mu)
    Phi = get_Phi(theta, beta)
    return H + R @ Phi @ T

def get_opt_Q_hat(H, T, R, pow_tx, pow_noise, theta: np.ndarray, theta_min, theta_max, beta_min, psi, mu, WF_loop):
    assert theta.ndim == 1 and theta.size == T.shape[0],\
        f"theta must be a vector with size of {T.shape[0]}. \nCurrent theta = {theta}"
    H_hat = get_H_hat(H, T, R, theta, beta_min, psi, mu)
    U, lambdaa, VH = np.linalg.svd(H_hat, full_matrices=False)
    S = lambdaa.size

    p, _, _, _ = water_filling(a=np.ones(S), b=np.ones(S), c=lambdaa**2/pow_noise, d=np.zeros(S),
                               f=np.ones(S), P=pow_tx, tol=WF_loop.tol_out, iter_lim=WF_loop.iterlim_out)
    Q_hat = VH.T.conj() @ np.diag(np.sqrt(p))
    rate = get_rate(pow_tx, pow_noise, H, T, R, theta, Q_hat, beta_min, psi, mu)
    return Q_hat, rate

def water_filling(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, f: np.ndarray, P: float, tol: float,
                  iter_lim: int):
    """
    Given the following optimization problem:
    min     -sum_{n=1}^N a_n log(b_n+ p_n c_n)
    s.t.    p_n >= d_n,
            f^T p = P,
    this function find vector p that solve the problem.

    Arguments:
    (1) a : 1D positive-entries tensor
    (2) b : 1D non-negative-entries tensor having the same size with a
    (3) c : 1D positive-entries tensor having the same size with a
    (4) d : 1D non-negative-entries tensor having the same size with a
    (5) f : 1D positive-entries tensor having the same size with a
    (6) P : positive scalar
    (7) tol : tolerance for the stopping criterion. It must be a positive scalar
    (8) iter_max : maximum allowed iteration for a stopping criterion. It must be a positive integer scalar.

    Outputs:
    (1) obj_val : the optimal value of the objective function
    (2) p : the optimal primal variable p
    (3) exec_time : the execution time
    (4) g_nu_trace : the recorded value of g (see the attached file) during iteration.
                     this variable may useful to plot the algorithm's convergence.
    """
    assert np.all(a > 0) and a.ndim == 1, f"Input a must be a vector with positive entries. \nCurrent input a = {a}"
    N = a.size
    assert np.all(b >= 0) and b.ndim == 1, f"Input b must be a vector with non-negative entries. \nCurrent input b = {b}"
    assert np.all(c > 0) and c.ndim == 1, f"Input c must be a vector with positive entries. \nCurrent input c = {c}"
    assert np.all(d >= 0) and d.ndim == 1, f"Input d must be a vector with non-negative entries. \nCurrent input d = {d}"
    assert np.all(f > 0) and f.ndim == 1, f"Input f must be a vector with positive entries. \nCurrent input f = {f}"
    assert P > 0, f"Input P must be positive scalar. \nCurrent input P = {P}"
    assert tol > 0, f"Input tol must be positive scalar. \nCurrent input tol = {tol}"
    assert iter_lim > 0, f"Input iter_liim must be integer positive. \nCurrent input iter_lim = {iter_lim}"
    assert np.dot(d, f) <= P, f"The constraint is not feasible. \nThe value of P = {P}. \nThe value of dot(d, f) = {np.dot(d, f)}."

    start_timer = time.time()
    iter_th = 0
    nu = (-np.dot(f, d) + P + np.dot(f, b/c+d)) / np.sum(a)  # Init nu
    # g_nu = 0
    g_nu_trace = np.array([])

    while iter_th <= iter_lim:
        # print(f"WF iter_th = {iter_th} -----------")
        g_nu = np.sum(f * np.maximum(nu*a/f - b/c - d, np.zeros(N))) + np.dot(f, d) - P  # ; print("type(g_nu) = ", type(g_nu))
        g_nu = np.expand_dims(g_nu, axis=0)
        g_nu_trace = np.concatenate((g_nu_trace, g_nu), axis=0)
        if np.abs(g_nu) < tol:
            break
        dg_dnu = np.sum(a * (nu*a/f - b/c - d >= 0))
        nu = nu - g_nu / dg_dnu
        iter_th += 1
    p = np.maximum(nu*a/f - b/c, d)
    exec_time = time.time() - start_timer
    obj_val = -np.sum(a * np.log(b + p*c))
    return p, obj_val, exec_time, g_nu_trace

# ----------- Phase-Shift Optimization ----------- #
def get_opt_theta(H: np.ndarray, T: np.ndarray, R: np.ndarray, Q_hat: np.ndarray, pow_tx: float, pow_noise: float,
                  theta_prev: np.ndarray, theta_min: np.ndarray, theta_max: np.ndarray, beta_min: np.ndarray,
                  psi: np.ndarray, mu: np.ndarray, ps_loop: cf.loop_crit):
    assert theta_prev.ndim == 1, f"theta_prev must be a 1D vector. \nCurrent theta_prev.ndim = {theta_prev.ndim}"
    assert np.all(theta_prev >= theta_min) and np.all(theta_prev <= theta_max), \
        f"All values of theta_prev must be between {theta_min} and {theta_max}. \nCurrent theta_prev = {theta_prev}"
    Nr = H.shape[0]
    Nt = H.shape[1]
    M = T.shape[0]

    Q = Q_hat @ Q_hat.T.conj()

    # Eigen-Decomposition of Q
    sigma_Q, U_Q = np.linalg.eig(Q)  # ; print(f"sigma_Q = {sigma_Q}")
    Sigma_Q = np.diag(sigma_Q)
    # print(f"Q original  = {Q}")
    # print(f"Q SVD = {U_Q @ Sigma_Q @ U_Q.T.conj()}")

    # Calculating H_prime and T_prime
    H_prime = H @ U_Q @ Sigma_Q**(1/2)  # ; print(f"Sigma_Q**(1/2) = {Sigma_Q**(1/2)}")
    T_prime = T @ U_Q @ Sigma_Q**(1/2)

    theta = theta_prev.copy()

    iter_th = 0
    while iter_th < ps_loop.iterlim_out:
        iter_th += 1  # ; print(f"get_opt_theta() iter_th = {iter_th} -----------")
        for m in range(M):
            A_m, B_m, C_m = get_mat_ABC_m(H_prime, T_prime, R, pow_noise, theta, theta_min, theta_max, beta_min, psi, mu, Nt, Nr, M, m)

            # update theta using (17)
            beta_m = get_beta(theta[m], beta_min, psi, mu)
            eigvals = np.linalg.eigvals(np.linalg.inv(A_m + beta_m ** 2 * B_m) @ (beta_m * C_m))
            lambda_m = eigvals[np.argmax(np.abs(eigvals))]
            theta[m] = -np.angle(lambda_m)
            theta[m] = cf.bound_angles(theta[m], theta_min, theta_max)
            # print(f"   eigvals = {eigvals}")
            # print(f"   lambda_m = {lambda_m}")
            # rate_v2 = get_rate_v2(A_m, B_m, C_m, theta[m], beta_min, psi, mu, Nr)  # print(f"   rate_v2 = {rate_v2}")
    rate = get_rate(pow_tx, pow_noise, H, T, R, theta, Q_hat, beta_min, psi, mu)  # ; print(f"   rate = {rate}")
    return theta, rate

def get_mat_ABC_m(H_prime: np.ndarray, T_prime: np.ndarray, R, pow_noise, theta, theta_min, theta_max, beta_min, psi, mu,
                  Nt: int, Nr: int, M: int, m: int):
    assert np.array_equal(H_prime.shape, (Nr, Nt)),\
        f"Input H_prime must be a matrix of shape ({Nt}, {Nt}). \nCurrent H_prime shape = {H_prime.shape}"
    assert np.array_equal(T_prime.shape, (M, Nt)),\
        f"Input T_prime must be a matrix of shape ({M}, {Nt}). \nCurrent T_prime shape = {T_prime.shape}"
    assert m < M, f"the value of m = {m} must be smaller than M = {M}"
    # assert np.all(theta > theta_min) and np.all(theta <= theta_max), \
    #     f"All values of theta must be between {theta_min} and {theta_max}. \nCurrent theta = {theta}"

    beta = get_beta(theta, beta_min, psi, mu)
    Phi = get_Phi(theta, beta)

    X = H_prime.copy()
    for i in range(M):
        if i != m:
            X += Phi[i, i] * R[:, (i,)] @ T_prime[(i,), :]
    A_m = np.eye(Nr) + X @ X.T.conj() / pow_noise
    B_m = R[:, (m,)] @ T_prime[(m,), :] @ (T_prime[(m,), :]).T.conj() @ (R[:, (m,)]).T.conj() / pow_noise
    C_m = R[:, (m,)] @ T_prime[(m,), :] @ X.T.conj() / pow_noise

    return A_m, B_m, C_m

# ----------- Common function ----------- #
def get_rate(pow_tx: float, pow_noise: float, H: np.ndarray, T: np.ndarray, R: np.ndarray, theta: np.ndarray,
             Q_hat: np.ndarray, beta_min: np.ndarray, psi: np.ndarray, mu: np.ndarray):
    """ Equation (3) """
    assert pow_tx > 0, \
        f"Input pow_tx must be positive - It's in watt unit. If your input is still in dBm or dBw,convert it to watt first." \
        f"\nCurrent input pow_tx = {pow_tx}"
    assert pow_noise > 0, \
        f"Input pow_noise must be positive - It's in watt unit. If your input is still in dBm or dBw,convert it to watt first." \
        f"\nCurrent input pow_noise = {pow_noise}"
    assert H.ndim == 2 and T.ndim == 2 and R.ndim == 2 and Q_hat.ndim == 2 and theta.ndim == 1, \
        f"The dimension of channels H, T, R, and Q_hat must be 2, and that of theta must be 1. \nCurrent H.ndim = {H.ndim}." \
        f"\nCurrent T.ndim = {T.ndim}. \nCurrent R.ndim = {R.ndim}. \nCurrent Q_hat.ndim = {Q_hat.ndim}." \
        f"\nCurrent theta.ndim = {theta.ndim}"
    # assert theta_min.ndim <= 1 and theta_min.size == 1, \
    #     f"Input theta_min must be a scalar with ndim <= 1.\nCurrent theta_min = {theta_min}."
    # assert theta_max.ndim <= 1 and theta_max.size == 1, \
    #     f"Input theta_max must be a scalar with ndim <= 1.\nCurrent theta_max = {theta_max}."
    assert beta_min.ndim <= 1 and beta_min.size == 1 and np.all(beta_min >= 0), \
        f"Input beta_min must be a scalar with ndim <= 1 whose value is non-negative. \nCurrent beta_min = {beta_min}."
    assert psi.ndim <= 1 and psi.size == 1 and np.all(psi > -np.pi) and np.all(psi <= np.pi), \
        f"Input psi mus be a scalar with ndim <= 1 whose value is between -pi and pi. \nCurrent psi = {psi}"
    assert mu.ndim <= 1 and mu.size == 1 and np.all(mu >= 0), \
        f"Input alpha must be a scalar with ndim <= 1 whose value is non-negative. \nCurrent alpha = {mu}."
    # assert np.all(theta > theta_min) and np.all(theta <= theta_max), \
    #     f"All values of theta must be between {theta_min} and {theta_max}. \nCurrent theta = {theta}"
    Nr = H.shape[0]
    Nt = H.shape[1]
    M = T.shape[0]
    assert T.shape[1] == Nt, \
        f"The value of H.shape[1] and and T.shape[1] must be the same. \nCurrent H.shape[1] = {Nt}." \
        f"\nCurrent T.shape[1] = {T.shape[1]}."
    assert R.shape[0] == Nr, \
        f"The value of H.shape[0] and R.shape[0] must be the same. \nCurrent value of H.shape[0] = {Nr}." \
        f"\nCurrent value of R.shape[0] = {R.shape[0]}."
    assert R.shape[1] == M, \
        f"The value of T.shape[0] and R.shape[1] must be the same. \nCurrent value of T.shape[0] = {M}." \
        f"\nCurrent value of R.shape[1] = {R.shape[1]}."
    assert theta.size == M, f"The size of theta must be {M}. \nCurrent theta size = {theta.size}"
    assert Q_hat.shape[0] == Nt, f"THe value of Q_hat.shape[0] mus be {Nt}. \nCurrent Q_hat.shape[0] = {Q_hat.shape[0]}."
    assert np.linalg.norm(Q_hat, ord='fro')**2 <= pow_tx * 1.01,\
        f"The square of the Frobenius norm of Q_hat must be less than Tx power." \
        f"\nCurrent square of Frobenius norm of Q_hat = {np.linalg.norm(Q_hat, ord='fro')**2}. \nCurrent pow_tx = {pow_tx}"

    H_hat = get_H_hat(H, T, R, theta, beta_min, psi, mu)
    return np.abs(np.log2(np.linalg.det(np.eye(Nr) + H_hat @ Q_hat @ Q_hat.T.conj() @ H_hat.T.conj() / pow_noise)))

def get_rate_v2(A_m: np.ndarray, B_m: np.ndarray, C_m: np.ndarray, theta_m: np. ndarray, beta_min: np.ndarray,
                psi: np.ndarray, mu: np.ndarray, Nr: int):
    """ Equation (14) """
    assert np.array_equal(A_m.shape, (Nr, Nr)) and np.array_equal(B_m.shape, (Nr, Nr)) and np.array_equal(C_m.shape, (Nr, Nr)),\
        f"The shape of A_m, B_m, and C_m must be {Nr}x{Nr}. \nCurrent input A_m shape = {A_m.shape}." \
        f"\nCurrent input B_m shape = {B_m.shape}. \nCurrent input C_m shape = {C_m.shape}"
    assert theta_m.size == 1, f"The size of theta_m must be 1. \nCurrent theta_m.size = {theta_m.size}"
    # assert np.all(theta_m > theta_min) and np.all(theta_m <= theta_max), \
    #     f"All values of theta_m must be between {theta_min} and {theta_max}. \nCurrent theta_m = {theta_m}"
    beta_m = get_beta(theta_m, beta_min, psi, mu)
    return np.abs(np.log2(np.linalg.det(A_m + beta_m**2 * B_m + np.exp(1j*theta_m) * beta_m * C_m + np.exp(-1j*theta_m) * beta_m * C_m.T.conj())))
