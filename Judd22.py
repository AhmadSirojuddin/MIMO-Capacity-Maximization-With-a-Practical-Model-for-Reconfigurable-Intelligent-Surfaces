import numpy as np
import time
import common_functions as cf
import warnings

# ------- Functions to be Revoked in other files ------- #
def main(F: np.ndarray, G: np.ndarray, H: np.ndarray, tx_pow: np.ndarray, noise_pow: np.ndarray, theta_init: np.ndarray,
         theta_min: np.ndarray, theta_max: np.ndarray, dsm_loop: cf.loop_crit, wf_loop: cf.loop_crit):
    assert F.ndim == 2 and G.ndim == 2 and H.ndim == 2,\
        f"The dimension number of F, G, and H must be 2. \nCurrent F.ndim = {F.ndim}. \nCurrent G.ndim = {G.ndim}." \
        f"\nCurrent H.ndim = {H.ndim}"
    assert tx_pow.ndim <= 1 and tx_pow.size == 1 and tx_pow > 0,\
        f"Input tx_pow must be scalar positive. Current tx_pow = {tx_pow}. \nIf hour tx_pow is in dB, convert it first" \
        f"to the linear scale."
    assert noise_pow.size == 1 and noise_pow > 0,\
        f"Input noise_pow must be scalar positive. Current noise_pow = {noise_pow}. \nIf your noise_pow is in dB," \
        f"convert it first to the linear scale."
    assert theta_min.ndim <= 1 and theta_min.size == 1, \
        f"Input theta_min must be a scalar with ndim <= 1.\nCurrent theta_min = {theta_min}."
    assert theta_max.ndim <= 1 and theta_max.size == 1, \
        f"Input theta_max must be a scalar with ndim <= 1.\nCurrent theta_max = {theta_max}."
    assert theta_init.ndim == 1 and np.all(theta_init > theta_min) and np.all(theta_init <= theta_max), \
        f"All values of theta_init_temp must be a 1D vector whose entries are between-pi and pi." \
        f"\nCurrent theta_init_temp = {theta_init}"
    K = F.shape[1]
    L = F.shape[0]
    N = H.shape[0]
    assert H.shape[1] == K,\
        f"F.shape[1] and H.shape[1] must be the same. \nCurrent F.shape[1] = {K}. \nCurrent H.shape[1] = {H.shape[1]}."
    assert G.shape[0] == L,\
        f"F.shape[0] and G.shape[0] must be the same. \nCurrent F.shape[0] = {L}. \nCurrent G.shape[0] = {G.shape[0]}."
    assert G.shape[1] == N,\
        f"H.shape[0] and G.shape[1] must be the same. \nCurrent H.shape[0] = {N}. \nCurrent G.shape[1] = {G.shape[1]}."

    # ----------- Finding a constant multiplier for tackling the numerical stability issue ----------- #
    smpl_num = N * 4
    psi_smpl = np.zeros(shape=smpl_num)
    for smpl_th in range(smpl_num):
        theta_rnd = 2 * np.pi * (np.random.rand(N) - 0.5)
        psi_smpl[smpl_th] = get_psi(F, G, H, theta_rnd)
    psi_smpl_mean = np.mean(psi_smpl); print(f"psi_smpl_mean = {psi_smpl_mean}")
    stable = 1 / psi_smpl_mean; print(f"stable = {stable}")

    start_time = time.time()

    # ----------- the DSM algorithm ----------- #
    _, theta, niter, _ = dsm(F, G, H, theta_init, dsm_loop, N, stable)
    theta = cf.bound_angles(theta, theta_min, theta_max)

    H_eff = get_H_eff(F, G, H, theta)
    mat_U, lambdaa, mat_VH = np.linalg.svd(H_eff, full_matrices=False)
    U = np.linalg.matrix_rank(H_eff)

    # ----------- the water filling algorithm ----------- #
    ones = np.ones(U)
    zeros = np.zeros(U)
    c = (np.square(lambdaa) ** 2) / noise_pow
    p, obj_val, _, _ = water_filling(ones, ones, c, zeros, ones, tx_pow, wf_loop.tol_out, wf_loop.iterlim_out)
    T = mat_VH.T.conj() @ np.diag(np.sqrt(p))

    time_exec = time.time() - start_time

    return obj_val, T, theta, time_exec

def dsm(F, G, H, theta_init, dsm_loop, N, stable):
    theta = theta_init.copy()
    psi = get_psi(F, G, H, theta)
    psi_trace = np.expand_dims(psi, axis=0)
    GFH = G.T.conj() @ F @ H.T.conj()
    GG = G.T.conj() @ G
    HH = H @ H.T.conj()
    iter_th = 0
    err = np.inf
    while np.abs(err) > dsm_loop.tol_out and iter_th < dsm_loop.iterlim_out:
        iter_th += 1
        for n in range(N):
            theta[n] = get_theta_dsm(GG[n, :], HH[:, n], GFH[n, n], theta, n)
        psi = get_psi(F, G, H, theta)
        err = (psi - psi_trace[-1]) * stable
        psi_trace = np.concatenate((psi_trace, np.expand_dims(psi, axis=0)), axis=0)
    if iter_th >= dsm_loop.iterlim_out:
        warnings.warn(f"WARNING! 'iter_th' reaches the limit (={dsm_loop.iterlim_out}), the iteration may not converge")
    niter = iter_th
    return psi, theta, niter, psi_trace

def get_psi(F, G, H, theta):
    Phi = np.diag(np.exp(1j*theta))
    return (np.linalg.norm(F + G @ Phi @ H, ord='fro'))**2

def get_theta_dsm(g_n, h_n, f_nn, theta, n):
    return np.angle(f_nn + g_n.dot(np.exp(1j*theta) * h_n) - g_n[n] * np.exp(1j*theta[n]) * h_n[n])

def get_H_eff(F, G, H, theta):
    return F + G @ np.diag(np.exp(1j*theta)) @ H

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
