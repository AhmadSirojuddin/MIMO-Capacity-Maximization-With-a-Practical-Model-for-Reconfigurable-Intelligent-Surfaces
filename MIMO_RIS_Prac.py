import numpy as np
import matplotlib.pyplot as plt
import common_functions as cf
import time
import warnings
import cvxpy as cp

# ----------- Main Functions ----------- #
def main(pow_tx: float, pow_noise: float, Hd: np.ndarray, G: np.ndarray, Hr: np.ndarray, theta_init: np.ndarray,
         theta_min: np.ndarray, theta_max: np.ndarray, beta_min: np.ndarray, theta_hat: np.ndarray, alpha: np.ndarray,
         main_loop: cf.loop_crit, GA_loop: cf.loop_crit, WF_loop: cf.loop_crit, step_reduct: np.ndarray):
    """

    Args:
        pow_tx    : Total TX power, in watt. If your data is in dBm, convert it using dBm_to_watt(dBm) in common_functions.py
        pow_noise : noise power, in watt. If your data is in dBm, convert it using dBm_to_watt(dBm) in common_functions.py
        Hd        : Direct channel from the BS to the UE
        G         : Channel from the BS to RIS
        Hr        : Channel from RIS to the UE
        theta_init: Initial guess for optimal phase shift
        theta_min : the maximum value of the phase shift. It corresponds to the practical model of RIS. The details can be
                    found in:
                    S. Abeywickrama, R. Zhang, Q. Wu, and C. Yuen, “Intelligent reflecting surface: Practical phase shift
                    model and beamforming optimization,” IEEE Trans. Commun., vol. 68, no. 9, pp. 5849–5863, Sep. 2020.
                    The value of theta_min, along with other parameters regarding the practical model RIS can be obtained
                    after executing method get_RIS_coeff()
        theta_max : the minimum value of the phase shift. See the explanation for theta_min.
        beta_min  : one of practical RIS model parameter. See the explanation for theta_min.
        theta_hat : one of practical RIS model parameter. See the explanation for theta_min.
        alpha     : one of practical RIS model parameter. See the explanation for theta_min.
        main_loop : parameters controlling the loop behavior in Alg. 1 line 1-24. It contains the maximum iteration number
                    and the tolerance for stopping the iteration
        GA_loop   : parameters controlling the loop behavior in Alg. 1 line 8-23. Its explanation is the same as main_loop
        WF_loop   : parameters controlling the loop behavior in Alg. 1 line 4-6. Its explanation is the same as main_loop
        step_reduct: step-size reduction parameter, denoted as 'c' in the manuscript.

    Returns:
        rate       : The capacity (SE) of the considered MIMO System
        T          : The optimal precoder T at the multi-antenna BS
        theta      : The optimal RIS phase-shifts
        time_exec  : The execution time of the code
        rate_trace : The series values of rates during the iteration as the counter variable 'i' increases. It may be
                     useful if we want to plot the convergence behavior of the Alg. 1.

    """
    assert pow_tx > 0,\
        f"Input pow_tx must be positive - It's in watt unit. If your input is still in dBm or dBw,convert it to watt first." \
        f"\nCurrent input pow_tx = {pow_tx}"
    assert pow_noise > 0, \
        f"Input pow_noise must be positive - It's in watt unit. If your input is still in dBm or dBw,convert it to watt first." \
        f"\nCurrent input pow_noise = {pow_noise}"
    assert Hd.ndim == 2 and G.ndim == 2 and Hr.ndim == 2 and theta_init.ndim == 1,\
        f"The dimension of Hd, G, and Hr must be 2, and theta_init must be 1. \nCurrent Hd.ndim = {Hd.ndim}. \nCurrent G.ndim = {G.ndim}." \
        f"\nCurrent Hr.ndim = {Hr.ndim}. \nCurrent theta_init.ndim = {theta_init.ndim}"
    assert theta_min.ndim <= 1 and theta_min.size == 1,\
        f"Input theta_min must be a scalar with ndim <= 1.\nCurrent theta_min = {theta_min}."
    assert theta_max.ndim <= 1 and theta_max.size == 1, \
        f"Input theta_max must be a scalar with ndim <= 1.\nCurrent theta_max = {theta_max}."
    assert beta_min.ndim <= 1 and beta_min.size == 1 and np.all(beta_min >= 0),\
        f"Input beta_min must be a scalar with ndim <= 1 whose value is non-negative. \nCurrent beta_min = {beta_min}."
    assert theta_hat.ndim <= 1 and theta_hat.size == 1 and np.all(theta_hat > -np.pi) and np.all(theta_hat <= np.pi),\
        f"Input theta_hat mus be a scalar with ndim <= 1 whose value is between -pi and pi. \nCurrent theta_hat = {theta_hat}"
    assert alpha.ndim <= 1 and alpha.size == 1 and np.all(alpha >= 0),\
        f"Input alpha must be a scalar with ndim <= 1 whose value is non-negative. \nCurrent alpha = {alpha}."
    assert step_reduct.ndim <= 1 and step_reduct.size == 1 and np.all(step_reduct > 0) and np.all(step_reduct < 1),\
        f"Input step_reduct must be a scalar with ndim <= 1 whose value is between 0 and 1." \
        f"\nCurrent step_reduct = {step_reduct}"
    assert np.all(theta_init > theta_min) and np.all(theta_init <= theta_max),\
        f"All values of theta_init must be between-pi and pi. \nCurrent theta_init = {theta_init}"
    K = Hd.shape[0]
    M = Hd.shape[1]
    N = G.shape[0]
    assert G.shape[1] == M,\
        f"The value of Hd.shape[1] and G.shape[1] must be the same.\nCurrent Hd.shape[1] = {Hd.shape[1]}." \
        f"\nCurrent G.shape[1] = {G.shape[1]}."
    assert Hr.shape[0] == K,\
        f"The value of Hd.shape[0] and Hr.shape[0] must be the same. \nCurrent Hd.shape[0] = {Hd.shape[0]}." \
        f"\nCurrent Hr.shape[0] = {Hr.shape[0]}"
    assert Hr.shape[1] == N,\
        f"The value of Hr.shape[1] and G.shape[0] must be the same. \nCurrent Hr.shape[1] = {Hr.shape[1]}." \
        f"\nCurrent G.shape[0] = {G.shape[0]}"
    assert theta_init.size == N, f"The size of theta_init must be {N}. \nCurrent theta_init size = {theta_init.size}"

    time_start = time.time()

    rate_diff = np.inf
    iter_th = 0
    theta = theta_init.copy()
    rate_trace = np.array([])  # Dummy
    rate = 0  # Dummy
    T = 0  # Dummy
    while rate_diff > main_loop.tol_out and iter_th < main_loop.iterlim_out:
        iter_th += 1  # ; print(f"main_loop iter_th = {iter_th} ---------------------")

        # Optimization w.r.t. Precoder T
        T, rate = get_opt_T(Hd, G, Hr, pow_tx, pow_noise, theta, theta_min, theta_max, beta_min, theta_hat, alpha, WF_loop)
        # print(f"   rate_judd (opt T) = {rate}")

        # Optimization w.r.t. phase shift
        # theta_prev, rate_prev, exec_time, rate_trace
        theta, rate, _, rate_trace_GA = get_theta_opt_GD(pow_tx, pow_noise, Hd, G, Hr, theta, T, theta_min, theta_max, beta_min, theta_hat, alpha, GA_loop, step_reduct)
        # print(f"   rate_trace_GA = {rate_trace_GA}")
        # print(f"   rate_judd (opt theta) = {rate}")

        if iter_th == 1:
            rate_trace = np.expand_dims(rate, axis=0)
        else:
            rate_trace = np.concatenate((rate_trace, np.expand_dims(rate, axis=0)), axis=0)
            rate_diff = np.abs(rate_trace[-1] - rate_trace[-2])
    # print(f"   rate_trace (main, judd) = {rate_trace}")
    # print(f"   rate_diff (main) = {rate_diff}")

    if iter_th >= main_loop.iterlim_out:
        warnings.warn(f"WARNING! 'iter_th' reaches the limit (={iter_th}), the iteration may not converge")
    time_exec = time.time() - time_start
    return rate, T, theta, time_exec, rate_trace

def main_v2(pow_tx: float, pow_noise: float, Hd: np.ndarray, G: np.ndarray, Hr: np.ndarray, theta_init: np.ndarray,
            theta_min: np.ndarray, theta_max: np.ndarray, beta_min: np.ndarray, theta_hat: np.ndarray, alpha: np.ndarray,
            main_loop: cf.loop_crit, GA_loop: cf.loop_crit, WF_loop: cf.loop_crit, step_reduct: np.ndarray):
    assert pow_tx > 0,\
        f"Input pow_tx must be positive - It's in watt unit. If your input is still in dBm or dBw,convert it to watt first." \
        f"\nCurrent input pow_tx = {pow_tx}"
    assert pow_noise > 0, \
        f"Input pow_noise must be positive - It's in watt unit. If your input is still in dBm or dBw,convert it to watt first." \
        f"\nCurrent input pow_noise = {pow_noise}"
    assert Hd.ndim == 2 and G.ndim == 2 and Hr.ndim == 2 and theta_init.ndim == 1,\
        f"The dimension of Hd, G, and Hr must be 2, and theta_init must be 1. \nCurrent Hd.ndim = {Hd.ndim}. \nCurrent G.ndim = {G.ndim}." \
        f"\nCurrent Hr.ndim = {Hr.ndim}. \nCurrent theta_init.ndim = {theta_init.ndim}"
    assert theta_min.ndim <= 1 and theta_min.size == 1,\
        f"Input theta_min must be a scalar with ndim <= 1.\nCurrent theta_min = {theta_min}."
    assert theta_max.ndim <= 1 and theta_max.size == 1, \
        f"Input theta_max must be a scalar with ndim <= 1.\nCurrent theta_max = {theta_max}."
    assert beta_min.ndim <= 1 and beta_min.size == 1 and np.all(beta_min >= 0),\
        f"Input beta_min must be a scalar with ndim <= 1 whose value is non-negative. \nCurrent beta_min = {beta_min}."
    assert theta_hat.ndim <= 1 and theta_hat.size == 1 and np.all(theta_hat > -np.pi) and np.all(theta_hat <= np.pi),\
        f"Input theta_hat mus be a scalar with ndim <= 1 whose value is between -pi and pi. \nCurrent theta_hat = {theta_hat}"
    assert alpha.ndim <= 1 and alpha.size == 1 and np.all(alpha >= 0),\
        f"Input alpha must be a scalar with ndim <= 1 whose value is non-negative. \nCurrent alpha = {alpha}."
    assert step_reduct.ndim <= 1 and step_reduct.size == 1 and np.all(step_reduct > 0) and np.all(step_reduct < 1),\
        f"Input step_reduct must be a scalar with ndim <= 1 whose value is between 0 and 1." \
        f"\nCurrent step_reduct = {step_reduct}"
    assert np.all(theta_init > theta_min) and np.all(theta_init <= theta_max),\
        f"All values of theta_init must be between-pi and pi. \nCurrent theta_init = {theta_init}"
    K = Hd.shape[0]
    M = Hd.shape[1]
    N = G.shape[0]
    assert G.shape[1] == M,\
        f"The value of Hd.shape[1] and G.shape[1] must be the same.\nCurrent Hd.shape[1] = {Hd.shape[1]}." \
        f"\nCurrent G.shape[1] = {G.shape[1]}."
    assert Hr.shape[0] == K,\
        f"The value of Hd.shape[0] and Hr.shape[0] must be the same. \nCurrent Hd.shape[0] = {Hd.shape[0]}." \
        f"\nCurrent Hr.shape[0] = {Hr.shape[0]}"
    assert Hr.shape[1] == N,\
        f"The value of Hr.shape[1] and G.shape[0] must be the same. \nCurrent Hr.shape[1] = {Hr.shape[1]}." \
        f"\nCurrent G.shape[0] = {G.shape[0]}"
    assert theta_init.size == N, f"The size of theta_init must be {N}. \nCurrent theta_init size = {theta_init.size}"

    time_start = time.time()

    rate_diff = np.inf
    iter_th = 0
    theta = theta_init.copy()
    rate_trace = np.array([])  # Dummy
    rate = 0  # Dummy
    T = 0  # Dummy
    while rate_diff > main_loop.tol_out and iter_th < main_loop.iterlim_out:
        iter_th += 1  # ; print(f"main_loop iter_th = {iter_th} ---------------------")

        # Optimization w.r.t. Precoder T
        T, rate = get_opt_T(Hd, G, Hr, pow_tx, pow_noise, theta, theta_min, theta_max, beta_min, theta_hat, alpha, WF_loop)
        # print(f"   rate_judd (opt T) = {rate}")

        # Optimization w.r.t. phase shift
        # theta_prev, rate_prev, exec_time, rate_trace
        theta, rate, _, rate_trace_GA = get_theta_opt_GD(pow_tx, pow_noise, Hd, G, Hr, theta, T, theta_min, theta_max, beta_min, theta_hat, alpha, GA_loop, step_reduct)
        # print(f"   rate_trace_GA = {rate_trace_GA}")
        # print(f"   rate_judd (opt theta) = {rate}")

        if iter_th == 1:
            rate_trace = np.expand_dims(rate, axis=0)
        else:
            rate_trace = np.concatenate((rate_trace, np.expand_dims(rate, axis=0)), axis=0)
            rate_diff = np.abs(rate_trace[-1] - rate_trace[-2])
    # print(f"   rate_trace (main, judd) = {rate_trace}")
    # print(f"   rate_diff (main) = {rate_diff}")

    if iter_th >= main_loop.iterlim_out:
        warnings.warn(f"WARNING! 'iter_th' reaches the limit (={iter_th}), the iteration may not converge")
    time_exec = time.time() - time_start
    return rate, T, theta, time_exec, rate_trace

# ----------- RIS Curve Fitting ----------- #
def get_RIS_coeff(C_var: np.ndarray, resist: np.ndarray, freq: float, iter_lim: int, tol: float, show_plot=False):
    """
    This function follows the practical RIS equation model proposed in 10.1109/TCOMM.2020.3001125 equation (5)
    parameters:

    returns:
    # theta_min: np.ndarray of shape the same of that of resist
    # theta_max: np.ndarray of shape the same of that of resist
    # beta_min: np.ndarray of shape the same of that of resist
    # theta_hat: np.ndarray of shape the same of that of resist
    # alpha: np.ndarray of shape the same of that of resist
    """
    assert C_var.ndim == 1, f"Input C_var must have dimension = 1. \nYour C_var.ndim = {C_var.ndim}"
    assert np.all(C_var > 1e-13) and np.all(C_var < 1e-11),\
        f"Current model only support the range of C from 0.47 pF until 2.35 pF. \nYour current C_var = {C_var}"
    assert np.all(resist >= 0), f"Input resist must be positive. \nYour input resist = {resist}"
    assert resist.ndim == 1, f"Input resist must have dimension = 1. \nYour resist.ndim = {resist.ndim}"
    assert freq == 2.4e9, f"This model only valid of frequency = 2.4 GHz. \nYour input 'freq'={freq}"
    assert iter_lim > 0, f"Input iter_lim must be positive. \nYour input iter_lim = {iter_lim}"
    assert tol > 0, f"Input tol must be positive. \nYour input tol = {tol}"

    L1 = 2.5e-9
    L2 = 0.7e-9
    Z0 = 377
    C_var = np.expand_dims(C_var, axis=0)  # ; print(f"C_var = {C_var}")
    resist = np.expand_dims(resist, axis=1)  # ; print(f"resist = {resist}")

    # Calculating the RIS Refl. coeffs. given the values by inputs
    Z = get_RIS_impedance(C_var, resist, L1, L2, freq)  # ; print(f"Z.shape = {Z.shape}")
    beta_set, theta_set = get_RIS_refl_coeff(Z, Z0)
    # print(f"beta_set.shape = {beta_set.shape}")
    # print(f"theta_set.shape = {theta_set.shape}")
    # print(f"beta_set = {beta_set}")
    # print(f"theta_set = {theta_set}")
    theta_min = np.min(theta_set, axis=1)  # ; print(f"theta_min = {theta_min}")
    theta_max = np.max(theta_set, axis=1)  # ; print(f"theta_max = {theta_max}")

    # Get beta_min and theta_hat
    beta_min = np.min(beta_set, axis=1, keepdims=True)
    # beta_min_idx = np.argmin(beta_set, axis=1, keepdims=True)  # ; print(f"beta_min_idx = {beta_min_idx}")
    theta_hat = np.abs(np.take_along_axis(theta_set, np.argmin(beta_set, axis=1, keepdims=True), axis=1) + np.pi / 2)
    # print(f"beta_min = {beta_min}")
    # print(f"theta_hat (pi rad)= {theta_hat / np.pi}")

    # Get alpha, MSE minimization using GD
    alpha = np.zeros(resist.size)
    for resist_th in range(resist.size):
        # print(f"resist_th = {resist_th} ----------------------")
        iter_th = 0  # iteration index
        alpha_prev = np.array([[5.]])  # initialization
        alpha_now = 0
        # beta_eq = get_RIS_ampl(theta_set[[resist_th], :], beta_min[[resist_th], :], theta_hat[[resist_th], :], alpha_prev)
        # print(f"   beta_eq shape = {beta_eq.shape}")
        mse_prev = get_mse(beta_set[[resist_th], :],
                           get_beta(theta_set[[resist_th], :], beta_min[[resist_th], :], theta_hat[[resist_th], :],
                                    alpha_prev))  # ; print(f"   mse_prev = {mse_prev}")
        h = 1e-3
        err = np.inf
        step_size = 0  # dummy
        while iter_th < iter_lim and np.abs(err) > tol and mse_prev > tol:
            iter_th += 1  # ; print(f"   iter_th = {iter_th} ----------------------")
            grad = (get_mse(beta_set[[resist_th], :],
                            get_beta(theta_set[[resist_th], :], beta_min[[resist_th], :], theta_hat[[resist_th], :],
                                     alpha_prev + h)) - mse_prev) / h
            if iter_th == 1:
                step_size = 1 / grad  # ; print(f"      step_size = {step_size}")
            iter_th2 = 0
            err = np.inf
            while iter_th2 < iter_lim and np.abs(err) > tol:
                iter_th2 += 1  # ; print(f"      iter_th2 = {iter_th2} ---------------------")
                alpha_now = alpha_prev - step_size * grad  # ; print(f"      alpha_now = {alpha_now}")
                mse_now = get_mse(beta_set[[resist_th], :],
                                  get_beta(theta_set[[resist_th], :], beta_min[[resist_th], :],
                                           theta_hat[[resist_th], :], alpha_now))
                # print(f"      mse_now = {mse_now}")
                err = mse_now - mse_prev  # ; print(f"      err = {err}")
                if err < 0:
                    alpha_prev = alpha_now.copy()  # ; print(f"      UPDATE alpha_prev = {alpha_prev}")
                    mse_prev = mse_now.copy()
                    break
                else:
                    step_size *= 0.7  # ; print(f"      REDUCING step_size = {step_size}")
        alpha[resist_th] = alpha_now
    # Plot, if requested
    if show_plot:
        _, plot_ampl_ang = plt.subplots()
        for resist_th in range(resist.size):
            plot_ampl_ang.plot(theta_set[resist_th, :], beta_set[resist_th, :])
        plot_ampl_ang.set_ylim(bottom=0.)
        plt.show()

    return theta_min, theta_max, beta_min.squeeze(), theta_hat.squeeze(), alpha

def get_RIS_impedance(C_var: np.ndarray, resist: float | np.ndarray, L1: float, L2: float, freq: float):
    omega = 2 * np.pi * freq
    term1 = 1j * omega * L1
    term2 = 1j * omega * L2 + 1/(1j*omega*C_var) + resist
    Z = term1 * term2 / (term1 + term2)
    return Z

def get_RIS_refl_coeff(Z, Z0: float):
    phi = (Z - Z0) / (Z + Z0)
    return np.abs(phi), np.angle(phi)

def get_beta(theta: np.ndarray, beta_min: np.ndarray, theta_hat: np.ndarray, alpha: np.ndarray):
    return (1 - beta_min) * ((np.sin(theta - theta_hat) + 1) / 2)**alpha + beta_min

def get_mse(in1: np.ndarray, in2: np.ndarray):
    return ((in1 - in2)**2).mean()

def get_Phi(theta: np.ndarray, beta: np.ndarray):
    assert theta.ndim == 1,\
        f"Input theta must be a 1D-array with ndim=1.\nCurrent theta.ndim = {theta.ndim}."
    assert np.array_equal(theta.shape, beta.shape),\
        f"The shape of theta and beta must be the same." \
        f"\nCurrent input theta.shape = {theta.shape}. \nCurrent input beta.shape = {beta.shape}."
    return np.diag(beta * np.exp(1j*theta))

# ----------- Precoder Optimization ----------- #
def get_H_eff(Hd, G, Hr, theta, beta_min, theta_hat, alpha):
    beta = get_beta(theta, beta_min, theta_hat, alpha)
    Phi = get_Phi(theta, beta)
    return Hd + Hr @ Phi @ G

def get_opt_T(Hd, G, Hr, pow_tx, pow_noise, theta, theta_min, theta_max, beta_min, theta_hat, alpha, WF_loop):
    assert theta.ndim == 1 and theta.size == G.shape[0],\
        f"theta must be a vector with size of {G.shape[0]}. \nCurrent theta = {theta}"
    H_eff = get_H_eff(Hd, G, Hr, theta, beta_min, theta_hat, alpha)
    U, lambdaa, VH = np.linalg.svd(H_eff, full_matrices=False)
    # print(f"lambdaa = {lambdaa}")
    S = lambdaa.size  # ; print(f"S = {S}")

    # print(f"H_eff original = {H_eff}")
    # print(f"H_eff after SVD = {U @ np.diag(lambdaa) @ VH}")

    # Finding optimal power allocation using the Water-Filling Algorithm
    p, obj_val, exec_time, _ = water_filling(a=np.ones(S), b=np.ones(S), c=lambdaa**2/pow_noise, d=np.zeros(S),
                                             f=np.ones(S), P=pow_tx, tol=WF_loop.tol_out, iter_lim=WF_loop.iterlim_out)
    # print(f"power allocation (WF) = {p}")
    # print(f"total power in p = {np.sum(p)}")
    # print(f"obj_val (WF) = {obj_val}")
    # print(f"exec_time (WF) = {exec_time}")

    # p_cvx, obj_val_cvx, exec_time_cvx = water_filling_cvxpy(a=np.ones(S), b=np.ones(S), c=lambdaa**2/pow_noise, d=np.zeros(S), f=np.ones(S), P=pow_tx)
    # print(f"power allocation (CVX) = {p_cvx.value}")
    # print(f"obj_val (CVX) = {obj_val_cvx}")
    # print(f"exec_time (CVX) = {exec_time_cvx}")

    # print(f"power of VH = {np.linalg.norm(VH, ord='fro')**2}")
    T = VH.T.conj() @ np.diag(np.sqrt(p))
    # print(f"Confirming the power of T = {np.linalg.norm(T, ord='fro')**2}")
    # print(f"Confirming that SVD diagonalizes channel matrix = {U.T.conj() @ H_eff @ T}")

    rate = get_rate(pow_tx, pow_noise, Hd, G, Hr, theta, T, beta_min, theta_hat, alpha)
    return T, rate

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

def water_filling_cvxpy(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, f: np.ndarray, P: float):
    start = time.time()
    N = a.size
    p = cp.Variable(N)
    objective = cp.Minimize(-cp.sum(cp.multiply(a, cp.log(b + cp.multiply(p, c)))))
    constraints = [p >= d, cp.sum(cp.multiply(f, p)) == P]
    prob = cp.Problem(objective, constraints)
    obj_val = prob.solve()
    time_exec = time.time() - start
    return p, obj_val, time_exec

# ----------- Phase-Shift Optimization ----------- #
def get_theta_opt_GD(pow_tx, pow_noise, Hd, G, Hr, theta, T, theta_min, theta_max, beta_min, theta_hat, alpha, GA_loop, step_reduct):
    assert T.ndim == 2, f"The dimension numbers of T must be 2. \nCurrent T.ndim = {T.ndim}"
    assert np.linalg.norm(T, ord='fro') ** 2 <= pow_tx * 1.01, \
        f"The square of the Frobenius norm of T must be less than Tx power." \
        f"\nCurrent square of Frobenius norm of T = {np.linalg.norm(T, ord='fro') ** 2}. \nCurrent pow_tx = {pow_tx}"
    start_time = time.time()
    K = Hd.shape[0]
    iter_th_out = 0
    rate_diff = np.inf
    theta_prev = theta.copy()  # ; print(f"theta_prev (pi rad) = {theta_prev/np.pi}")
    rate_prev = get_rate(pow_tx, pow_noise, Hd, G, Hr, theta_prev, T, beta_min, theta_hat, alpha)
    # print(f"rate_prev (init) = {rate_prev}")
    rate_trace = np.expand_dims(rate_prev, axis=0)  # ; print(f"rate_trace = {rate_trace}")
    step_size_init = 1  # dummy to remove warning

    while np.abs(rate_diff) > GA_loop.tol_out and iter_th_out < GA_loop.iterlim_out:
        iter_th_out += 1  # ; print(f"iter_th_out (get_theta_opt_GD) = {iter_th_out} -----------")
        drate_dtheta = get_drate_dtheta(pow_noise, Hd, G, Hr, theta_prev, T, beta_min, theta_hat, alpha, K)  # ; print(f"drate_dtheta = {drate_dtheta}")
        if iter_th_out == 1:
            step_size = (0.25 * np.pi) / np.max(np.abs(drate_dtheta))
            step_size_init = step_size.copy()  # ; print(f"step_size_init = {step_size_init}")
        iter_th_in = 0
        step_size = step_size_init.copy()  # ; print(f"step_size = {step_size}")
        while iter_th_in < GA_loop.iterlim_in and np.abs(rate_diff) > GA_loop.tol_in:
            iter_th_in += 1  # ; print(f"   iter_th_in (get_theta_opt_GD) = {iter_th_in} -----------")
            # TO DO : Give a warning when iter_th_in reaches limit
            if iter_th_in >= GA_loop.iterlim_in:
                warnings.warn(f"WARNING! 'iter_th_in' reaches the limit (={GA_loop.iterlim_in}), the iteration may not converge")
            theta_now = theta_prev + step_size * drate_dtheta  # ; print(f"   theta_now (pi rad) = {theta_now/np.pi}")
            theta_now = cf.normalize_angles(theta_now)  # ; print(f"   theta_now normalized (pi rad) = {theta_now/np.pi}")
            theta_now = cf.bound_angles(theta_now, theta_min, theta_max)  # ; print(f"   theta_now bounded (pi rad) = {theta_now/np.pi}")
            rate_now = get_rate(pow_tx, pow_noise, Hd, G, Hr, theta_now, T, beta_min, theta_hat, alpha)  # ; print(f"   rate_now = {rate_now}")
            rate_diff = rate_now - rate_prev  # ; print(f"   rate_diff = {rate_diff}")
            if rate_diff > 0:
                # print(f"      rate_diff is LARGER than previous")
                theta_prev = theta_now.copy()
                rate_prev = rate_now.copy()
                rate_trace = np.concatenate((rate_trace, np.expand_dims(rate_now, axis=0)))
                break
            else:
                # print(f"      rate_diff is SMALLER than previous")
                step_size = step_size * step_reduct  # ; print(f"      step_size = {step_size}")
    exec_time = time.time() - start_time
    return theta_prev, rate_prev, exec_time, rate_trace
    # drate_dtheta = get_drate_dtheta(pow_noise, Hd, G, Hr, theta, T, beta_min, theta_hat, alpha, Nr)
    # print(f"drate_dtheta (equation) = {drate_dtheta}")
    # drate_dtheta = get_drate_dtheta_limit(pow_tx, pow_noise, Hd, G, Hr, theta, T, theta_min, theta_max, beta_min, theta_hat, alpha)
    # print(f"drate_dtheta (limit) = {drate_dtheta}")

def get_drate_dtheta(pow_noise, Hd, G, Hr, theta, T, beta_min, theta_hat, alpha, K: int):
    H_eff = get_H_eff(Hd, G, Hr, theta, beta_min, theta_hat, alpha)
    A = np.eye(K) + H_eff @ T @ T.T.conj() @ H_eff.T.conj() / pow_noise
    X = T @ T.T.conj() @ H_eff.T.conj() @ np.linalg.inv(A) @ Hr
    # np.diag(A @ B) is equal to np.sum(A * B.T, axis=1)
    return (1 / np.log(2) / pow_noise) * 2 * np.real(get_dphi_dtheta(theta, beta_min, theta_hat, alpha) * np.sum(G*X.T, axis=1))

def get_drate_dtheta_limit(pow_tx, pow_noise, Hd, G, Hr, theta, T, theta_min, theta_max, beta_min, theta_hat, alpha):
    increment = 1e-6
    N = theta.size
    drate_dtheta = np.zeros(N)
    rate1 = get_rate(pow_tx, pow_noise, Hd, G, Hr, theta, T, beta_min, theta_hat, alpha)
    for n in range(N):
        theta_temp = theta.copy()
        theta_temp[n] += increment
        rate2 = get_rate(pow_tx, pow_noise, Hd, G, Hr, theta_temp, T, beta_min, theta_hat, alpha)
        drate_dtheta[n] = (rate2 - rate1) / increment
    return drate_dtheta

def get_dphi_dtheta(theta: np.ndarray, beta_min: np.ndarray, theta_hat: np.ndarray, alpha: np.ndarray):
    assert theta.ndim == 1, f"Input theta must have dimension of 1. \nCurrent theta.ndim = {theta.ndim}"
    return get_dbeta_dtheta(theta, beta_min, theta_hat, alpha) * np.exp(1j*theta)\
        + get_beta(theta, beta_min, theta_hat, alpha) * 1j * np.exp(1j*theta)

def get_dbeta_dtheta(theta: np.ndarray, beta_min: np.ndarray, theta_hat: np.ndarray, alpha: np.ndarray):
    return (1-beta_min) * alpha / 2**alpha * (np.sin(theta - theta_hat) + 1)**(alpha-1) * np.cos(theta - theta_hat)

# ----------- Phase-Shift Optimization (sub-optimal) ----------- #
# def get_theta_subopt_GD(Hd, G, Hr, theta, )

# ----------- Common function ----------- #
def get_rate(pow_tx: float, pow_noise: float, Hd: np.ndarray, G: np.ndarray, Hr: np.ndarray, theta: np.ndarray,
             T: np.ndarray, beta_min: np.ndarray, theta_hat: np.ndarray, alpha: np.ndarray):
    assert pow_tx > 0, \
        f"Input pow_tx must be positive - It's in watt unit. If your input is still in dBm or dBw,convert it to watt first." \
        f"\nCurrent input pow_tx = {pow_tx}"
    assert pow_noise > 0, \
        f"Input pow_noise must be positive - It's in watt unit. If your input is still in dBm or dBw,convert it to watt first." \
        f"\nCurrent input pow_noise = {pow_noise}"
    assert Hd.ndim == 2 and G.ndim == 2 and Hr.ndim == 2 and theta.ndim == 1 and T.ndim, \
        f"The dimension of Hd, G, Hr, and T must be 2, and theta_init must be 1. \nCurrent Hd.ndim = {Hd.ndim}. \nCurrent G.ndim = {G.ndim}." \
        f"\nCurrent Hr.ndim = {Hr.ndim}. \nCurrent theta_init.ndim = {theta.ndim}/ \nCurrent T.ndim = {T.ndim}"
    # assert theta_min.ndim <= 1 and theta_min.size == 1, \
    #     f"Input theta_min must be a scalar with ndim <= 1.\nCurrent theta_min = {theta_min}."
    # assert theta_max.ndim <= 1 and theta_max.size == 1, \
    #     f"Input theta_max must be a scalar with ndim <= 1.\nCurrent theta_max = {theta_max}."
    assert beta_min.ndim <= 1 and beta_min.size == 1 and np.all(beta_min >= 0), \
        f"Input beta_min must be a scalar with ndim <= 1 whose value is non-negative. \nCurrent beta_min = {beta_min}."
    assert theta_hat.ndim <= 1 and theta_hat.size == 1 and np.all(theta_hat > -np.pi) and np.all(theta_hat <= np.pi), \
        f"Input theta_hat mus be a scalar with ndim <= 1 whose value is between -pi and pi. \nCurrent theta_hat = {theta_hat}"
    assert alpha.ndim <= 1 and alpha.size == 1 and np.all(alpha >= 0), \
        f"Input alpha must be a scalar with ndim <= 1 whose value is non-negative. \nCurrent alpha = {alpha}."
    # assert np.all(theta >= theta_min) and np.all(theta <= theta_max), \
    #     f"All values of theta must be between {theta_min} and {theta_max}. \nCurrent theta = {theta}"
    K = Hd.shape[0]
    M = Hd.shape[1]
    N = G.shape[0]
    assert G.shape[1] == M, \
        f"The value of Hd.shape[1] and G.shape[1] must be the same.\nCurrent Hd.shape[1] = {Hd.shape[1]}" \
        f"\nCurrent G.shape[1] = {G.shape[1]}."
    assert Hr.shape[0] == K, \
        f"The value of Hd.shape[0] and Hr.shape[0] must be the same. \nCurrent Hd.shape[0] = {Hd.shape[0]}" \
        f"\nCurrent Hr.shape[0] = {Hr.shape[0]}"
    assert Hr.shape[1] == N, \
        f"The value of Hr.shape[1] and G.shape[0] must be the same. \nCurrent Hr.shape[1] = {Hr.shape[1]}" \
        f"\nCurrent G.shape[0] = {G.shape[0]}"
    assert theta.size == N, f"The size of theta must be {N}. \nCurrent theta size = {theta.size}"
    assert T.shape[0] == M, f"The value of T.shape[0] must be {M}. \nCurrent T.shape[0] = {T.shape[0]}"
    assert np.linalg.norm(T, ord='fro')**2 <= pow_tx * 1.01,\
        f"The square of the Frobenius norm of T must be less than Tx power." \
        f"\nCurrent square of Frobenius norm of T = {np.linalg.norm(T, ord='fro')**2}. \nCurrent pow_tx = {pow_tx}"

    H_eff = get_H_eff(Hd, G, Hr, theta, beta_min, theta_hat, alpha)
    return np.abs(np.log2(np.linalg.det(np.eye(K) + H_eff @ T @ T.T.conj() @ H_eff.T.conj() / pow_noise)))

