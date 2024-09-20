import numpy as np
from typing import Union
from collections import namedtuple
import shelve
import os
import warnings

loop_crit = namedtuple('loop_crit', ['iterlim_out', 'iterlim_in', 'tol_out', 'tol_in'], defaults=[0, 0, np.inf, np.inf])

def dBm_to_watt(dBm):
    return 10**(dBm/10 - 3)

def dB_to_lin(dB):
    return 10**(dB/10)

def LoS_far_field(rx_size: list, tx_size: list, ant_spc_rx: list, ant_spc_tx: list, AoA: list, AoD: list, wave_len: float):
    """
    Generates a MIMO channel for LOS_ far-field model. It is expressed as
    H = a_r(phi, theta) @ a_t(phi, theta)**H
    Each element of output has a magnitude value of 1 - this function does not take into account the effect of
    antennas gain and path loss
    Inputs:
    (1) rx_size : The antenna structure at RX, either ULA or UPA. If ULA is considered, input is a list with single
                  element representing the number of antennas in the horizontal direction. If UPA structure is
                  considered, input is 1D-list of length 2 where the 0th and 1st elements are the number of cols (horizontal)
                  and rows (vertical) of the UPA structure, respectively. All elements must be positive integers.
    (2) tx_size : The antenna structure at TX, either ULA or UPA. If ULA is considered, input is a list with single
                  element representing the number of antennas in the horizontal direction. If UPA structure is
                  considered, input is 1D-list of length 2 where the 0th and 1st elements are the number of cols (horizontal)
                  and rows (vertical) of the UPA structure, respectively. All elements must be positive integers.
    (3) ant_spc_rx : The spacing between antennas at RX, either ULA or UPA. If ULA in considered, input is a list with
                     single element representing the distance between antennas in the horizontal direction. If UPA is
                     considered, input is 1D-list of length 2 where the 0th and 1st elements are the distances between two
                     adjacent antennas in the horizontal and vertical direction, respectively. All elements must be real
                     and positive.
    (4) ant_spc_tx : The spacing between antennas at TX, either ULA or UPA. If ULA in considered, input is a list with
                     single element representing the distance between antennas in the horizontal direction. If UPA is
                     considered, input is 1D-list of length 2 where the 0th and 1st elements are the distances between two
                     adjacent antennas in the horizontal and vertical direction, respectively. All elements must be real
                     and positive.
    (5) AoA : The angle of arrival at RX, either ULA or UPA. If ULA is considered, input is a list with single
              element representing the azimuth (horizontal) angle of the beam. If UPA is considered, input is 1D-list of
              length 2 where the 0th and 1st elements are the azimuth (horizontal) and elevation (vertical) angles,
              respectively. All elements must be real and in range from 0 to pi (radian unit). Angle of 90 degrees means
              the beam is perpendicular to the antenna array.
    (6) AoD : The angle of departure at TX, either ULA or UPA. If ULA is considered, input a list with single
              element representing the azimuth (horizontal) angle of the beam. If UPA is considered, input is 1D-list of
              length 2 where the 0th and 1st elements are the azimuth (horizontal) and elevation (vertical) angles,
              respectively. All elements must be real and in range from 0 to pi (radian unit). Angle of 90 degrees means
              the beam is perpendicular to the antenna array.
    (7) wave_len : wave length of the RF signal, in meter. It must be real, scalar, and positive.
    """

    assert all(isinstance(var, int) for var in rx_size) and all(var > 0 for var in rx_size),\
        f"All elements in rx_size must be positive integer. Your rx_size = {rx_size}"
    assert all(isinstance(var, int) for var in tx_size) and all(var > 0 for var in tx_size), \
        f"All elements in tx_size must be positive integer. Your tx_size = {tx_size}"
    assert all(var > 0 for var in ant_spc_rx),\
        f"All elements in ant_spc_rx must be positive. Your ant_spc_rx = {ant_spc_rx}"
    assert all(var > 0 for var in ant_spc_tx), \
        f"All elements in ant_spc_tx must be positive. Your ant_spc_tx = {ant_spc_tx}"
    assert all(var > 0 for var in AoA) and all(var <= np.pi for var in AoA),\
        f"All elements in AoA must be between 0 and pi. Your AoA = {AoA}"
    assert all(var > 0 for var in AoD) and all(var <= np.pi for var in AoD), \
        f"All elements in AoD must be between 0 and pi. Your AoD = {AoD}"
    assert wave_len > 0,\
        f"Input wave_len must be positive. Your input wave_len = {wave_len}"
    assert rx_size.__len__() == ant_spc_rx.__len__() and rx_size.__len__() == AoA.__len__(),\
        f"There is a inconsistency between the length of rx_size, ant_spc_rx, and AoA." \
        f"\nLength of rx_size = {rx_size.__len__()}" \
        f"\nLength of ant_spc_rx = {ant_spc_rx.__len__()}" \
        f"\nLength of AoA = {AoA.__len__()}"
    assert tx_size.__len__() == ant_spc_tx.__len__() and tx_size.__len__() == AoD.__len__(), \
        f"There is a inconsistency between the length of tx_size, ant_spc_tx, and AoD." \
        f"\nLength of tx_size = {tx_size.__len__()}" \
        f"\nLength of ant_spc_tx = {ant_spc_tx.__len__()}" \
        f"\nLength of AoD = {AoD.__len__()}"

    if rx_size.__len__() == 1:
        rx_size.append(1)
        ant_spc_rx.append(0)
        AoA.append(np.pi/2)
    if tx_size.__len__() == 1:
        tx_size.append(1)
        ant_spc_tx.append(0)
        AoD.append(np.pi/2)
    # print(f'rx_size = {rx_size}')
    # print(f'ant_spc_rx = {ant_spc_rx}')
    # print(f'AoA = {AoA}')
    # print(f'tx_size = {tx_size}')
    # print(f'ant_spc_tx = {ant_spc_tx}')
    # print(f'AoD = {AoD}')

    # Receiver Structure
    rx_response_az = np.exp(1j * 2 * np.pi * ant_spc_rx[0] / wave_len * np.sin(AoA[1]) * np.cos(AoA[0])
                            * np.arange(1, rx_size[0]+1))
    rx_response_el = np.exp(1j * 2 * np.pi * ant_spc_rx[1] / wave_len * np.cos(AoA[1]) * np.arange(1, rx_size[1]+1))
    rx_response = np.kron(rx_response_az, rx_response_el).reshape((-1, 1))

    # Transmitter Structure
    tx_response_az = np.exp(1j * 2 * np.pi * ant_spc_tx[0] / wave_len * np.sin(AoD[1]) * np.cos(AoD[0])
                            * np.arange(1, tx_size[0]+1))
    tx_response_el = np.exp(1j * 2 * np.pi * ant_spc_tx[1] / wave_len * np.cos(AoD[1]) * np.arange(1, tx_size[1]+1))
    tx_response = np.kron(tx_response_az, tx_response_el).reshape((1, -1))

    return rx_response @ tx_response

def path_loss_simplified(dist: Union[float, np.ndarray], path_loss_ref: float, path_loss_exp: float,
                         dist_ref: float = 1.):
    # first_pow_profile = path_loss_ref * ((dist / dist_ref) ** -path_loss_exp)
    """
    Args:
    # dist: Distance from Tx to Rx. It can either a scalar or 2D-np.ndarray representing the distances of
            each Tx-Rx antenna pairs. All input elements must be real and positive.
    # path_loss_ref: Path loss at 'dist_ref', in linear scale (not in dB). 'path_loss_ref' must be scalar, and positive.
    # path_loss_exp: Path loss exponent, must be scalar and non-negative
    # dist_ref: Reference distance at which 'path_loss_ref' is defined. 'dist_ref' must be scalar, and positive.

    Returns:

    """
    assert np.all(dist > 0), f"Input 'dist' must be positive. \nYour dist = {dist}"
    assert path_loss_ref > 0, f"Input 'path_loss_ref' must be positive. It is in linear scale, not dB." \
                              f"\nYour 'path_loss_ref' = {path_loss_ref}"
    assert path_loss_exp > 0, f"Input 'path_loss_exp' must be scalar positive." \
                              f"\nYour input 'path_loss_exp' = {path_loss_exp}"
    assert dist_ref > 0, f"Input 'dist_ref' must be scalar positive. \nYour input 'dist_ref' = {dist_ref}"

    return path_loss_ref * ((dist / dist_ref) ** -path_loss_exp)

def rich_scatter_ch(ch_size: list, LoS_component: np.ndarray = 0, Rician_fact: Union[float, np.ndarray] = 0,
                    path_loss: Union[float, np.ndarray] = 1, cir_power: np.ndarray = 0.):
    """
    This function outputs a CIR for the wideband MIMO channel realization. The first tap of CIR is based on Rician
    fading channel, whereas the rest taps is based on the Rayleigh fading channel.
    The first tap can be Rayleigh by ignoring input parameter LoS_component and Rician_fact.
    This function can also generate a Rayleigh model channel by ignoring input 'cir_power'.
    Args:
    # ch_size: 1D-list of length 2 where the 0th and 1st entries represent the antenna numbers in RX and TX, respectively
    # LoS_component: complex matrix of size 'ch_size' containing the LOS_ information of the channel. If it remains unfilled,
                     this function outputs Rayleigh-modeled channel realization. All entries in LoS_component must have
                     the magnitude of one.
    # Rician_fact: Rician factor is ratio of LoS and NLoS wireless power, it is in linear scale (not dB) in this code.
                   Rician factor with zero value means no LoS is contained in channel, while a high value of Rician factor
                   indicates the strong effect of LoS over NLoS. It can be a float scalar where the value is applied to
                   all MIMO channel elements, or np.ndarray of size 'ch_size' where each MIMO channel element has its
                   own Rician factor.
    # path_loss: path loss of the first CIR, in linear scale (not dB) representing the ratio of the received signal power
                 to the transmitted signal power. It can be a float scalar where the value is applied to all MIMO channel
                 elements, or np.ndarray of size 'ch_size' where each MIMO channel element has its own path gain.
    # cir_power: If wideband channel with multiple path is considered, the power of all path is specified here.
                       The entries in 'multipath_power' are relative power (not absolute power) to the first tap of CIR,
                       where the first tap of CIR is assumed to be one. (The absolute power of the first tap CIR is
                       specified in 'path_loss'. Given a channel with N-tap CIR, then 'multipath_power' is 1D np.ndarray
                       with size of N-1. All entries in 'multipath_power' must be positive.
    Returns:

    """
    assert ch_size.__len__() == 2,\
        f"The length of ch_size must be 2 where the 0th ad 1st elements represent the size of RX and TX, respectively." \
        f"Your ch_size = {ch_size}"
    assert all(isinstance(var, int) for var in ch_size) and all(var > 0 for var in ch_size), \
        f"All elements in ch_size must be positive integer. Your ch_size = {ch_size}"
    tol = 1e-6
    if type(LoS_component) == np.ndarray:
        assert np.all(np.abs(LoS_component) > 1 - tol) and np.all(np.abs(LoS_component) < 1 + tol), \
            f"Absolute value of all elements in LoS_component for far-field model in this code must be 1." \
            f"\nThe effect of path loss in not considered in this function."\
            f"\nCurrent absolute value of LoS_component = {np.abs(LoS_component)}"
        assert np.array_equal(LoS_component.shape, ch_size),\
            f"The size of LoS_component must be the same as ch_size." \
            f"\nYour LoS_component shape = {LoS_component.shape}." \
            f"\nYour ch_size = {ch_size}"

    assert np.all(Rician_fact >= 0),\
        f"Rician_fact must be non-negative, and it is in the linear scale (not dB)." \
        f"\nYour Rician_fact = {Rician_fact}"
    if type(Rician_fact) == np.ndarray:
        assert np.array_equal(Rician_fact.shape, ch_size), \
            f"The shape of Rician_fact must be the same as ch_size." \
            f"\nYour Rician_fact shape = {Rician_fact.shape}." \
            f"\nYour ch_size = {ch_size}"

    assert np.all(path_loss > 0),\
        f"path_gain must be non-negative, and it is in the linear scale (not dB)."\
        f"\nYour path_gain = {path_loss}"
    if type(path_loss) == np.ndarray:
        assert np.array_equal(path_loss.shape, ch_size),\
            f"The shape of path_gain must be the same as ch_size." \
            f"\nYour path_gain shape = {path_loss.shape}." \
            f"\nYour ch_size = {ch_size}"

    if type(cir_power) == np.ndarray:
        assert cir_power.ndim == 1 and np.all(cir_power > 0),\
            f"multipath_power must be a 1D array will positive entries, and the first element must be 1." \
            f"\nAll entries are in linear, not dB."\
            f"\nYour multipath_power = {cir_power}"

    NLoS_component = (np.random.randn(ch_size[0], ch_size[1]) + 1j*np.random.randn(ch_size[0], ch_size[1])) / np.sqrt(2)
    first_tap_ch = (path_loss ** 0.5) * (np.sqrt(Rician_fact / (1 + Rician_fact)) * LoS_component
                                         + np.sqrt(1 / (1 + Rician_fact)) * NLoS_component)
    if type(cir_power) != np.ndarray:
        return first_tap_ch
    else:
        ch = np.zeros(shape=(cir_power.size + 1, ch_size[0], ch_size[1]), dtype=np.complex128)
        ch[0, :, :] = first_tap_ch
        for tap_th in np.arange(start=1, stop=cir_power.size + 1):
            ch[tap_th, :, :] = ((cir_power[tap_th-1] * path_loss) ** 0.5)\
                               * (np.random.randn(ch_size[0], ch_size[1]) + 1j*np.random.randn(ch_size[0], ch_size[1]))\
                               / np.sqrt(2)
        return ch

def normalize_angles(angles):
    return -np.pi + np.mod(angles+np.pi, 2*np.pi)

def bound_angles(angles, theta_min, theta_max):
    return np.minimum(np.maximum(angles, theta_min), theta_max)

def manage_data(is_save: bool, is_reset: bool, filename: str, **xaxis_and_rates: np.ndarray):
    """
    return: d (dictionary)
    """
    # ------- Check the input ------- #
    idx = -1
    xaxis_len = 0  # dummy
    for key, value in xaxis_and_rates.items():
        idx += 1  # ; print('idx = ', idx, ' ---------------------')
        # print('key = ', key)
        # print('value = ', value)
        if idx == 0:
            assert value.ndim == 1,\
                ("The first keyword argument must be an array with ndim = 0. Your input = ", value)
            xaxis_len = value.size  # ; print('xaxis_len = ', xaxis_len)
        else:
            assert value.ndim == 3,\
                ("The keyword argument (except the first one) must be a tensor with ndim = 3. Your input with keyword '", key, "' = ", value)
            assert value.shape[2] == xaxis_len,\
                ("The first kwarg's size must be equal to the size of the rest kwarg dim 2. The first kwarg's size=", xaxis_len, ". size of input with keyword ", key, " dim 2 = ", value.shape[2])

    d = {}  # empty dictionary
    if not is_save:
        if is_reset:
            warnings.warn('WARNING! Resetting the stored data without saving new data is not allowed. \nThe keyword is_reset is now set to False')
        print('just bypass the input to the output ...........')
        for key, value in xaxis_and_rates.items():
            d[key] = value
            # print('key = ', key)
            # print('value = ', value)
    else:
        if is_reset or (not is_reset and not os.path.isfile(filename+'.dat')):
            # print(os.path.isfile(filename+'.dat'))
            print('create a new file ...........')
            my_shelf = shelve.open(filename, flag='n')  # Always create a new, empty database
            for key, value in xaxis_and_rates.items():
                # print('key = ', key)
                # print('value = ', value)
                d[key] = value  # just bypass the input to the output
                my_shelf[key] = value   # save the new data to the file
            my_shelf.close()
        else:
            print('read the existing file ...........')
            my_shelf = shelve.open(filename, flag='w')  # Open existing database for reading and writing

            idx = -1
            for key, value in xaxis_and_rates.items():
                idx += 1  # ; print('idx = ', idx, ' ---------------------')
                # print('key = ', key)
                # print('value = ', value)
                # print('my_shelf[key] = ', my_shelf[key])
                if idx == 0:
                    # check whether the stored x-axis and the new x-axis are the same
                    # if not torch.equal(my_shelf[key], value):
                    if not np.array_equal(my_shelf[key], value):
                        print('ERROR! The stored x-axis and the new one are different')
                        print('the stored x-axis = ', my_shelf[key])
                        print('the new x-axis = ', value)
                        raise ValueError('DATA MISMATCH')
                    continue
                # d[key] = torch.cat([my_shelf[key], value], dim=0)  # concat in the dimension of channel number
                d[key] = np.concatenate((my_shelf[key], value), axis=0)
                my_shelf[key] = d[key]  # Store back the concat data to the file
                # print('d[key] = ', d[key])
                # print('my_shelf[key] = ', my_shelf[key])
            my_shelf.close()
    return d
