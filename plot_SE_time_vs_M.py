"""
This code is the same as plot_SE_time_vs_N.py, except that the simulation parameters are set to be the same as those
is paper Wang21
This one is used in the paper
"""

import numpy as np
import common_functions as cf
import matplotlib.pyplot as plt
import MIMO_RIS_Prac as Pro
import Wang21 as Wang
import Judd22 as Judd
import Boyu20 as Boyu

# np.random.seed(13)

# ----------- RIS Properties ----------- #
resistance_set = np.array([0., 2.5])
freq = 2.4e9
C_var = np.linspace(0.47, 2.35, num=500)*1e-12

theta_min, theta_max, beta_min, theta_hat, alpha = Pro.get_RIS_coeff(C_var, resistance_set, freq, iter_lim=100, tol=1e-6)
print(f"theta_min = {theta_min}")
print(f"theta_max = {theta_max}")
print(f"beta_min = {beta_min}")
print(f"theta_hat = {theta_hat}")
print(f"alpha = {alpha}")
# _, plot_ampl_ang = plt.subplots()
# theta_set = np.linspace(-np.pi, np.pi, num=500)
# for resist_th in range(resistance_set.size):
#     plot_ampl_ang.plot(theta_set, Pro.get_beta(theta_set, beta_min[resist_th], theta_hat[resist_th], alpha[resist_th]))
# plot_ampl_ang.set_ylim(bottom=0.)
# plt.show()

# ----------- Channel Generation ----------- #
dist_Hd = 300
dist_G = np.sqrt(290**2 + 10**2)
dist_Hr = np.sqrt(10**2 + 10**2)
pl_ref = cf.dB_to_lin(-30)  # path loss at the reference distance
pl_exp_Hd = 3.5   # path loss exponent for Tx-to-Rx channel
pl_exp_G = 2.2  # path loss exponent for Tx-to-RIS channel
pl_exp_Hr = 2.8  # path loss exponent for RIS-to-Rx channel
Rician_fact = cf.dB_to_lin(10)
wave_len = 3e8/freq
dist_ant = wave_len/2
AoA = np.pi/2
AoD = np.pi/2

pl_Hd = cf.path_loss_simplified(dist=dist_Hd, path_loss_ref=pl_ref, path_loss_exp=pl_exp_Hd)
pl_G = cf.path_loss_simplified(dist=dist_G, path_loss_ref=pl_ref, path_loss_exp=pl_exp_G)
pl_Hr = cf.path_loss_simplified(dist=dist_Hr, path_loss_ref=pl_ref, path_loss_exp=pl_exp_Hr)
print(f"pl_Hd = {pl_Hd}")
print(f"pl_G = {pl_G}")
print(f"pl_Hr = {pl_Hr}")

# ----------- Plotting, Powers ----------- #
M_set = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)  # number of RIS elements
# M_set = (40,)
pow_tx = cf.dBm_to_watt(30)
pow_noise = cf.dBm_to_watt(-88)
ch_smpl_num = 10  # number of channel sampling
Nt = 4  # number of antennas at Tx
Nr = 4  # number of antennas at Rx

N_largest = max(M_set)  # ; print(f"N_largest = {N_largest}")

# ----------- Regarding The Proposed Model ----------- #
rate_pro = np.zeros((ch_smpl_num, resistance_set.size, M_set.__len__()))
time_pro = np.zeros((ch_smpl_num, resistance_set.size, M_set.__len__()))
pro_main_loop = cf.loop_crit(iterlim_out=20, iterlim_in=0, tol_out=1e-3, tol_in=np.inf)
pro_GA_loop = cf.loop_crit(iterlim_out=10, iterlim_in=50, tol_out=1e-3, tol_in=1e-3)
pro_wf_loop = cf.loop_crit(iterlim_out=100, iterlim_in=0, tol_out=1e-6, tol_in=np.inf)
pro_step_reduct = np.array([0.7])

# ----------- Regarding The Benchmark, Wang21 ----------- #
rate_wang = np.zeros((ch_smpl_num, resistance_set.size, M_set.__len__()))
time_wang = np.zeros((ch_smpl_num, resistance_set.size, M_set.__len__()))
wang_main_loop = cf.loop_crit(iterlim_out=20, iterlim_in=0, tol_out=1e-3, tol_in=np.inf)
wang_ps_loop = cf.loop_crit(iterlim_out=1, iterlim_in=0, tol_out=np.inf, tol_in=np.inf)
wang_wf_loop = cf.loop_crit(iterlim_out=100, iterlim_in=0, tol_out=1e-6, tol_in=np.inf)
L = 10  # number of initial random theta

# ----------- Regarding The Benchmark, Nema21 ----------- #
rate_nema = np.zeros((ch_smpl_num, resistance_set.size, M_set.__len__()))
time_nema = np.zeros((ch_smpl_num, resistance_set.size, M_set.__len__()))

# ----------- Regarding The Benchmark, Boyu ----------- #
rate_boyu = np.zeros((ch_smpl_num, resistance_set.size, M_set.__len__()))
time_boyu = np.zeros((ch_smpl_num, resistance_set.size, M_set.__len__()))
beta_boyu = np.array(1.)
boyu_admm_loop = cf.loop_crit(iterlim_out=50, iterlim_in=0, tol_out=1e-3, tol_in=np.inf)
boyu_wf_loop = cf.loop_crit(iterlim_out=100, iterlim_in=0, tol_out=1e-6, tol_in=np.inf)

# ----------- Regarding The Benchmark, Judd ----------- #
rate_judd = np.zeros((ch_smpl_num, resistance_set.size, M_set.__len__()))
time_judd = np.zeros((ch_smpl_num, resistance_set.size, M_set.__len__()))
judd_dsm_loop = cf.loop_crit(iterlim_out=20, iterlim_in=0, tol_out=1e-3, tol_in=np.inf)
judd_wf_loop = cf.loop_crit(iterlim_out=100, iterlim_in=0, tol_out=1e-6, tol_in=np.inf)

for ch_smpl_th in range(ch_smpl_num):
    # Big Channel Generation
    Hd = cf.rich_scatter_ch(ch_size=[Nr, Nt], path_loss=pl_Hd)  # ; print(f"Hd = {Hd}")

    G_LoS = cf.LoS_far_field(rx_size=[N_largest], tx_size=[Nt], ant_spc_rx=[dist_ant], ant_spc_tx=[dist_ant], AoA=[AoA],
                             AoD=[AoD], wave_len=wave_len)
    G_big = cf.rich_scatter_ch(ch_size=[N_largest, Nt], LoS_component=G_LoS, Rician_fact=Rician_fact, path_loss=pl_G)  # ; print(f"G_big = {G_big}")

    Hr_LoS = cf.LoS_far_field(rx_size=[Nr], tx_size=[N_largest], ant_spc_rx=[dist_ant], ant_spc_tx=[dist_ant], AoA=[AoA],
                              AoD=[AoD], wave_len=wave_len)
    Hr_big = cf.rich_scatter_ch(ch_size=[Nr, N_largest], LoS_component=Hr_LoS, Rician_fact=Rician_fact, path_loss=pl_Hr)  # ; print(f"Hr_big = {Hr_big}")

    # theta_init_big = (np.random.rand(N_largest) * np.abs(theta_max.min() - theta_min.max())) + theta_min.max()
    theta_init_big = (np.random.rand(L, N_largest) * np.abs(theta_max.min() - theta_min.max())) + theta_min.max()  # ; print(f"theta_init_big.shape = {theta_init_big.shape}")

    for N_th in range(M_set.__len__()):
        N = M_set[N_th]

        G = G_big[:N, :]  # ; print(f"G = {G}")
        Hr = Hr_big[:, :N]  # ; print(f"Hr = {Hr}")
        theta_init = theta_init_big[:, :N]  # ; print(f"theta_init.shape = {theta_init.shape}")

        for resist_th in range(resistance_set.__len__()):
            resist = resistance_set[resist_th]
            print(f"ch_smpl_th = {ch_smpl_th} | N = {N} | resist = {resist} =====================")

            # The Proposed Method
            rate_pro[ch_smpl_th, resist_th, N_th], _, _, time_pro[ch_smpl_th, resist_th, N_th], _ =\
                Pro.main(pow_tx, pow_noise, Hd, G, Hr, theta_init[0, :], theta_min[resist_th], theta_max[resist_th],
                         beta_min[resist_th], theta_hat[resist_th], alpha[resist_th], pro_main_loop, pro_GA_loop,
                         pro_wf_loop, pro_step_reduct)

            # The Benchmark, Wang21
            rate_wang[ch_smpl_th, resist_th, N_th], _, _, time_wang[ch_smpl_th, resist_th, N_th], _ =\
                Wang.main(pow_tx, pow_noise, Hd, G, Hr, theta_init, theta_min[resist_th], theta_max[resist_th],
                          beta_min[resist_th], theta_hat[resist_th], alpha[resist_th], wang_main_loop, wang_ps_loop,
                          wang_wf_loop)

            # The Benchmark, Judd22

            _, T_judd, theta_judd, time_judd[ch_smpl_th, resist_th, N_th] =\
                Judd.main(Hd, Hr, G, np.array(pow_tx), np.array(pow_noise), theta_init[0, :], theta_min[resist_th],
                          theta_max[resist_th], judd_dsm_loop, judd_wf_loop)
            rate_judd[ch_smpl_th, resist_th, N_th] =\
                Pro.get_rate(pow_tx, pow_noise, Hd, G, Hr, theta_judd, T_judd, beta_min[resist_th], theta_hat[resist_th],
                             alpha[resist_th])

            # The Benchmark, Boyu20
            _, T_boyu, theta_boyu, time_boyu[ch_smpl_th, resist_th, N_th] =\
                Boyu.main(np.array(pow_tx), beta_boyu, np.array(pow_noise), Hd.T.conj(), G, Hr.T.conj(), theta_init[0, :], theta_min[resist_th],
                          theta_max[resist_th], boyu_admm_loop, boyu_wf_loop)
            rate_boyu[ch_smpl_th, resist_th, N_th] =\
                Pro.get_rate(pow_tx, pow_noise, Hd, G, Hr, theta_boyu, T_boyu, beta_min[resist_th], theta_hat[resist_th],
                             alpha[resist_th])

            print(f"   rate_pro = {rate_pro[ch_smpl_th, resist_th, N_th]}")
            print(f"   rate_wang = {rate_wang[ch_smpl_th, resist_th, N_th]}")
            print(f"   rate_judd = {rate_judd[ch_smpl_th, resist_th, N_th]}")
            print(f"   rate_boyu = {rate_boyu[ch_smpl_th, resist_th, N_th]}")
            print(f"   ")
            print(f"   time_pro = {time_pro[ch_smpl_th, resist_th, N_th]}")
            print(f"   time_wang = {time_wang[ch_smpl_th, resist_th, N_th]}")
            print(f"   time_judd = {time_judd[ch_smpl_th, resist_th, N_th]}")
            print(f"   time_boyu = {time_boyu[ch_smpl_th, resist_th, N_th]}")

d = cf.manage_data(is_save=False, is_reset=False,
                   filename='D:/OneDrive - Business/Research/No5_Practical_RIS_MIMO/PythonCode/data_SE_time_vs_N',
                   N_set=np.array(M_set),
                   rate_pro=rate_pro, rate_wang=rate_wang, rate_boyu=rate_boyu, rate_judd=rate_judd, rate_nema=rate_nema,
                   time_pro=time_pro, time_wang=time_wang, time_boyu=time_boyu, time_judd=time_judd, time_nema=time_nema)
# for key, value in d.items():
#     print('key = ', key)
#     print('value = ', value)

# ------- Plot the data ------- #
line_colors = ((0, 0, 100 / 255), (0, 100 / 255, 0), (100 / 255, 0, 0), (100 / 255, 0, 100 / 255),
               (50 / 255, 50 / 255, 0), (0, 50 / 255, 50 / 255), (45 / 255, 14 / 255, 14 / 255))
line_styles = ('-', '--', ':', '-.')
line_markers = ('o', 'D', 's', '*', 'x', '+', '^', 'v', '>', '<')
markersize = 9.0
linewidth = 2.0
plt.rcParams["font.family"] = "Times New Roman"

_, plot_rate = plt.subplots(figsize=(7.2, 5.4))
_, plot_time = plt.subplots(figsize=(7.2, 5.4))

# Plotting Rate
for resist_th in range(resistance_set.size):
    # proposed method
    plot_rate.plot(M_set, d['rate_pro'].mean(axis=0)[resist_th, :], color=line_colors[0], marker=line_markers[0],
                   markersize=markersize, linestyle=line_styles[resist_th], linewidth=linewidth)
    # plot_time.semilogy(M_set, d['time_pro'].mean(axis=0)[resist_th, :], color=line_colors[0], marker=line_markers[0],
    #                    markersize=markersize, linestyle=line_styles[resist_th], linewidth=linewidth)
    plt.rcParams["font.family"] = "Times New Roman"

    # wang
    plot_rate.plot(M_set, d['rate_wang'].mean(axis=0)[resist_th, :], color=line_colors[1], marker=line_markers[1],
                   markersize=markersize, linestyle=line_styles[resist_th], linewidth=linewidth)
    # plot_time.semilogy(M_set, d['time_wang'].mean(axis=0)[resist_th, :], color=line_colors[1], marker=line_markers[1],
    #                    markersize=markersize, linestyle=line_styles[resist_th], linewidth=linewidth)
    plt.rcParams["font.family"] = "Times New Roman"

    # judd
    plot_rate.plot(M_set, d['rate_judd'].mean(axis=0)[resist_th, :], color=line_colors[2], marker=line_markers[2],
                   markersize=markersize, linestyle=line_styles[resist_th], linewidth=linewidth)
    # plot_time.semilogy(M_set, d['time_judd'].mean(axis=0)[resist_th, :], color=line_colors[2], marker=line_markers[2],
    #                    markersize=markersize, linestyle=line_styles[resist_th], linewidth=linewidth)
    plt.rcParams["font.family"] = "Times New Roman"

    # boyu
    plot_rate.plot(M_set, d['rate_boyu'].mean(axis=0)[resist_th, :], color=line_colors[3], marker=line_markers[3],
                   markersize=markersize+3, linestyle=line_styles[resist_th], linewidth=linewidth)
    # plot_time.semilogy(M_set, d['time_boyu'].mean(axis=0)[resist_th, :], color=line_colors[3], marker=line_markers[3],
    #                    markersize=markersize, linestyle=line_styles[resist_th], linewidth=linewidth)
    plt.rcParams["font.family"] = "Times New Roman"

# Plotting Time
plot_time.semilogy(M_set, d['time_pro'].mean(axis=(0, 1)), color=line_colors[0], marker=line_markers[0],
                   markersize=markersize, linestyle='-', linewidth=linewidth)
plot_time.semilogy(M_set, d['time_wang'].mean(axis=(0, 1)), color=line_colors[1], marker=line_markers[1],
                   markersize=markersize, linestyle='-', linewidth=linewidth)
plot_time.semilogy(M_set, d['time_judd'].mean(axis=(0, 1)), color=line_colors[2], marker=line_markers[2],
                   markersize=markersize, linestyle='-', linewidth=linewidth)
plot_time.semilogy(M_set, d['time_boyu'].mean(axis=(0, 1)), color=line_colors[3], marker=line_markers[3],
                   markersize=markersize+3, linestyle='-', linewidth=linewidth)


usedfont = {'fontname': 'Times New Roman', 'fontsize': 20}
plt.rcParams["axes.linewidth"] = 2.
plt.xticks(fontsize=20, fontfamily='Times New Roman')
plt.yticks(fontsize=20, fontfamily='Times New Roman')

plot_rate.set_title('SE versus N under channel realization = ' + str(d['rate_pro'].shape[0]))
plot_rate.set_xlabel('Number of RE in RIS', **usedfont)
plot_rate.set_ylabel('SE (bps/Hz)', **usedfont)
plot_rate.set_ylim(bottom=0.)
plot_rate.grid(color=(100/255, 100/255, 100/255), linewidth=1.5)

plot_time.set_title('Execution time versus N under channel realization = ' + str(d['time_pro'].shape[0]))
plot_time.set_xlabel('Number of RE in RIS', **usedfont)
plot_time.set_ylabel('Execution time (s)', **usedfont)
plot_time.grid(color=(100/255, 100/255, 100/255), linewidth=1.5)

plt.show()
