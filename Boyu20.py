import numpy as np
import time
import common_functions as cf
import warnings

def main(P: np.ndarray, beta: np.ndarray, noise_pow: np.ndarray, Hd_ch: np.ndarray, M_ch: np.ndarray, Hr_ch: np.ndarray,
         theta_init: np.ndarray, theta_min: np.ndarray, theta_max: np.ndarray, loop_admm: cf.loop_crit, loop_wf: cf.loop_crit):
    assert P.ndim == 0 and P > 0,\
        f"Input P must be a positive scalar. \nCurrent input P = {P}." \
        f"\nIf your P is in dBm or dBw, convert it first to watt."
    assert beta.ndim == 0 and 0 < beta <= 1, f"Input beta must be between 0 and 1. \nCurrent input beta = {beta}"
    assert noise_pow.ndim == 0 and noise_pow > 0,\
        f"Input noise_pow must be a positive scalar. \nCurrent input noise_pow = {noise_pow}." \
        f"\nIf your noise_pow is in dBm or dBw, convert it first to watt."
    assert Hd_ch.ndim == 2 and Hr_ch.ndim == 2 and M_ch.ndim == 2,\
        f"ndim for Hd_ch, Hr_ch, and M_ch must be 2.\nCurrent Hd_ch.ndim={Hd_ch.ndim}.\nCurrent Hr_ch.ndim={Hr_ch.ndim}." \
        f"\nCurrent M_ch.ndim={M_ch.ndim}"
    Nb = Hd_ch.shape[1]
    Nt = Hd_ch.shape[0]
    Nr = M_ch.shape[0]
    assert Hr_ch.shape[1] == Nb,\
        f"Hd_ch.shape[1] and Hr_ch.shape[1] must be the same;\nHd_ch.shape[1]={Nb}; \nHr_ch.shape[1]={Hr_ch.shape[1]};"
    assert M_ch.shape[1] == Nt,\
        f"Hd_ch.shape[0] and M_ch.shape[1] must be the same; \nHd_ch.shape[0]={Nt}; \nM_ch.shape[1]={M_ch.shape[1]};"
    assert Hr_ch.shape[0] == Nr,\
        f"M_ch.shape[0] and Hr_ch.shape[0] mus be the same; \nM_ch.shape[0]={Nr}; \nHr_ch.shape[0]={Hr_ch.shape[0]}"
    assert theta_min.ndim == 0 and -np.pi <= np.all(theta_min) <= np.pi,\
        f"theta_min must have a zero dimension and a value between -pi and pi. \nCurrent theta_min = {theta_min}"
    assert theta_max.ndim == 0 and -np.pi <= np.all(theta_max) <= np.pi, \
        f"theta_max must have a zero dimension and a value between -pi and pi. \nCurrent theta_max = {theta_max}"
    assert theta_init.ndim == 1 and theta_init.size == Nr and theta_min <= np.all(theta_init) <= theta_max,\
        f"theta_init must have ndim=1, size={Nr}, and values ranging from {theta_min} and {theta_max}." \
        f"\nCurrent theta_init = {theta_init} \nCurrent theta_init.ndim={theta_init.ndim}"

    # ----------- Finding a constant multiplier for tackling the numerical stability issue ----------- #
    smpl_num = Nr * 4
    obj_smpl = np.zeros(shape=smpl_num)
    for smpl_th in range(smpl_num):
        theta_rnd = 2 * np.pi * (np.random.rand(Nr) - 0.5)
        obj_smpl[smpl_th] = get_objective_8(Hr_ch, M_ch, Hd_ch, theta_rnd, beta)
    obj_smpl_mean = np.mean(obj_smpl)
    stable = 1 / obj_smpl_mean

    start_time = time.time()
    K_ch = M_ch @ Hd_ch
    T_hat, T_hat_eig_max = get_T_hat(beta, Hr_ch, M_ch, K_ch)

    # ----------- The ADMM Algorithm ----------- #
    rho = np.maximum(np.sqrt(2*T_hat_eig_max), T_hat_eig_max)
    x_init = np.exp(1j * np.concatenate((theta_init, np.array([0.])), axis=0))
    nu_init = np.random.randn(Nr+1) + 1j * np.random.randn(Nr+1)
    u_opt, x_opt, nu_opt, obj11_rec, k, time_exec = admm(T_hat, rho, x_init, nu_init, loop_admm, stable)

    _, Theta_opt = get_v_Theta(u_opt)
    H_eff = get_H_eff(Hr_ch, M_ch, Hd_ch, Theta_opt)
    mat_U, lambdaa, mat_VH = np.linalg.svd(H_eff, full_matrices=False)
    U = np.linalg.matrix_rank(H_eff)
    theta_opt = cf.bound_angles(np.diag(Theta_opt), theta_min, theta_max)

    # ----------- The Water-Filling Algorithm ----------- #
    ones = np.ones(U)
    zeros = np.zeros(U)
    c = (np.square(lambdaa) ** 2) / noise_pow
    p, obj_val, _, _ = water_filling(ones, ones, c, zeros, ones, P, loop_wf.tol_out, loop_wf.iterlim_out)

    precoder = mat_VH.T.conj() @ np.diag(np.sqrt(p))

    time_exec = time.time() - start_time

    return obj_val, precoder, theta_opt, time_exec

def get_T_hat(beta, Hr_ch, M_ch, K_ch):
    Nb = Hr_ch.shape[1]
    Nr = Hr_ch.shape[0]

    # Calculating T
    T_11 = 0.
    for nb in range(Nb):
        T_11 += np.diag(Hr_ch[:, nb].conj()) @ M_ch @ M_ch.T.conj() @ np.diag(Hr_ch[:, nb])
    T_11 = -(beta**2) * T_11

    T_12 = 0.
    for nb in range(Nb):
        T_12 += np.diag(Hr_ch[:, nb].conj()) @ K_ch[:, [nb]]
    T_12 = -beta * T_12

    T = np.concatenate((np.concatenate((T_11, T_12), axis=1), np.concatenate((T_12.T.conj(), np.array([[0.]])), axis=1)), axis=0)
    T_eig = np.sort(np.linalg.eigvals(T).real)
    T_hat = T - T_eig[0] * np.eye(Nr+1)
    T_hat_eig_max = np.max(np.linalg.eigvals(T_hat).real)
    return T_hat, T_hat_eig_max

def admm(T_hat: np.ndarray, rho: np.ndarray, x_prev: np.ndarray, nu_prev: np.ndarray, loop_admm: cf.loop_crit, stable):
    time_start = time.time()
    k = 0
    obj11_rec = get_objective_11a(T_hat, x_prev)
    err = np.inf
    _, Theta = get_v_Theta(np.exp(1j * np.angle(x_prev)))
    u_now = x_prev.copy()  # just dummy
    while k < loop_admm.iterlim_out and err > loop_admm.tol_out:
        k += 1
        u_now = np.exp(1j * np.angle(rho * x_prev - nu_prev))
        x_now = np.linalg.inv(rho * np.eye(T_hat.shape[0]) + T_hat) @ (rho * u_now + nu_prev)
        nu_now = T_hat @ x_now

        obj11_rec = np.concatenate((obj11_rec, get_objective_11a(T_hat, x_now)), axis=0)
        err = np.abs(obj11_rec[-1] - obj11_rec[-2]) * stable

        x_prev = x_now.copy()
        nu_prev = nu_now.copy()
    if k >= loop_admm.iterlim_out:
        warnings.warn(f"WARNING! 'iter_th' reaches the limit (={loop_admm.iterlim_out}), the iteration may not converge")
    time_exec = time.time() - time_start
    return u_now, x_prev, nu_prev, obj11_rec, k, time_exec

def get_objective_11a(T_hat, x):
    # return 0.5 * (T_hat @ x @ x.conj()).real
    return 0.5 * np.expand_dims((x.conj() @ T_hat @ x).real, axis=0)

def get_v_Theta(u: np.ndarray):
    v = u[0:-1] / u[-1]
    Theta = np.diag(v.conj())
    return v, Theta

def get_H_eff(Hr_ch: np.ndarray, M_ch: np.ndarray, Hd_ch:np.ndarray, Theta: np.ndarray):
    return M_ch.T.conj() @ Theta.T.conj() @ Hr_ch + Hd_ch

def water_filling(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, f: np.ndarray, P: float | np.ndarray,
                  tol: float, iter_lim: int):
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

def get_objective_8(Hr_ch, M_ch, Hd_ch, theta, beta):
    H_eff = get_H_eff(Hr_ch, M_ch, Hd_ch, np.diag(theta)*beta)
    return np.abs(np.trace(H_eff @ H_eff.T.conj()))
