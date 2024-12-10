#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import pickle
import os
from functools import partial
import p_tqdm
from pycbc.filter import sigma
from calcwf import *
from interpolating_match import *

def single_match(anomaly, e, chirp, q, f_low, f_match, n_gen, harm_ids, psd, distance, approximant='TEOBResumS'):

    # Calculate shifted f and frequency
    s_f_2pi = f_low - shifted_f(f_low, e, chirp2total(chirp, q), q)
    s_f = f_low - anomaly*s_f_2pi/(2*np.pi)
    s_e = shifted_e(s_f, f_low, e)

    # Generate and resize waveforms
    s = gen_wf(s_f, s_e, chirp2total(chirp, q), q, sample_rate, approximant=approximant, distance=distance)
    all_wfs = list(get_h([1]*n_gen, f_low, e, chirp2total(chirp, q), q, sample_rate, approximant=approximant))
    wf_hjs = all_wfs[1:len(harm_ids)+1]
    tlen = 2*(len(psd)-1)
    resized = resize_wfs([s, *wf_hjs], tlen=tlen)
    s = resized[0]
    wf_hjs = resized[1:]

    # Convert to frequency series
    t_start = s.sample_times[0]
    t_end = s.sample_times[-1]
    data = s.real().to_frequencyseries()
    h = {}
    for i, harm in enumerate(harm_ids):
        h[f'h{harm}'] = wf_hjs[i].real().to_frequencyseries()

    # Loop over detectors
    z = {}
    ifos = ['H1']
    for ifo in ifos:

        # Normalise waveform modes
        h_perp = {}
        for key in h.keys():
            h_perp[key] = h[key] / sigma(h[key], psd, low_frequency_cutoff=f_match, high_frequency_cutoff=psd.sample_frequencies[-1])

        # Calculate mode SNRs
        mode_SNRs, _ = calculate_mode_snr(data, psd, h_perp, t_start, t_end, f_match, h_perp.keys(), dominant_mode='h0')
        z[ifo] = mode_SNRs

    # Calculate network SNRs
    rss_snr, _ = network_mode_snr(z, ifos, h_perp.keys(), dominant_mode='h0')
    SNR_list = [rss_snr[f'h{harm}'] for harm in harm_ids]

    # Get match quantities
    h1_hn1 = np.linalg.norm(SNR_list[1:3])
    h1_hn1_frac = comb_harm_consistent(np.abs(SNR_list[:3]), np.angle(SNR_list[:3]), harms=[0,1,-1])
    h1_hn1_pc = h1_hn1_frac*np.abs(SNR_list[0])
    total_SNR = sigma(data, psd, low_frequency_cutoff=f_match, high_frequency_cutoff=psd.sample_frequencies[-1])
    ortho_SNR = np.sqrt(total_SNR**2 - np.abs(SNR_list[0])**2)

    # Save memory
    del s, data, all_wfs, wf_hjs, h, h_perp

    return *SNR_list, h1_hn1, h1_hn1_pc, total_SNR, ortho_SNR

def grid_data(vary_param, param_vals, MA_vals, n, n_gen, e, chirp, q, f_low, f_match, harm_ids, base_SNR, distance, approximant='TEOBResumS'):

    # Calculate parameters of longest waveform
    long_e, long_chirp, long_q = [e, chirp, q]
    if vary_param == 'e':
        long_e = np.min(param_vals)
    elif vary_param == 'chirp':
        long_chirp = np.min(param_vals)
    elif vary_param == 'q':
        long_q = np.max(param_vals)
    s_f_2pi = f_low - shifted_f(f_low, e, chirp2total(chirp, q), q)
    s_f_long = f_low - np.max(MA_vals)*s_f_2pi/(2*np.pi)
    s_e_long = shifted_e(s_f_long, f_low, e)

    # Generate longest possible waveform, and calculate psd
    long_wf = gen_wf(s_f_long, s_e_long, chirp2total(long_chirp, long_q), long_q, sample_rate, approximant, distance=distance)
    h_psd = timeseries.TimeSeries(list(long_wf)+[0], long_wf.delta_t, epoch=long_wf.start_time)
    psd = gen_psd(h_psd, f_low)

    # Turn anomaly and parameter values into meshgrid
    MA_mesh, param_mesh = np.meshgrid(MA_vals, param_vals)
    MA_mesh = list(np.array(MA_mesh).flatten())
    param_mesh = list(np.array(param_mesh).flatten())

    # Calculate all matches in parallel
    partial_single_match = partial(single_match, f_low=f_low, f_match=f_match, n_gen=n_gen, harm_ids=harm_ids, psd=psd, distance=distance, approximant=approximant)
    e_list = [e]*len(param_mesh)
    chirp_list = [chirp]*len(param_mesh)
    q_list = [q]*len(param_mesh)
    if vary_param == 'e':
        match_arr = np.array(p_tqdm.p_map(partial_single_match, MA_mesh, param_mesh, chirp_list, q_list))
    elif vary_param == 'chirp':
        match_arr = np.array(p_tqdm.p_map(partial_single_match, MA_mesh, e_list, param_mesh, q_list))
    elif vary_param == 'q':
        match_arr = np.array(p_tqdm.p_map(partial_single_match, MA_mesh, e_list, chirp_list, param_mesh))

    # Save memory
    del long_wf, psd

    # Put match arrays into appropriate dictionary keys
    matches = {}
    for i in range(n):
        matches[f'h{harm_ids[i]}'] = np.abs(match_arr[:,i].reshape(-1, len(MA_vals)))
        matches[f'h{harm_ids[i]}_phase'] = np.angle(match_arr[:,i].reshape(-1, len(MA_vals)))
    matches['h1_h-1'] = match_arr[:,n].reshape(-1, len(MA_vals))
    matches['h1_h-1_pc'] = match_arr[:,n+1].reshape(-1, len(MA_vals))
    matches['total_SNR'] = match_arr[:,n+2].reshape(-1, len(MA_vals))
    matches['ortho_SNR'] = match_arr[:,n+3].reshape(-1, len(MA_vals))

    # Add parameter keys
    matches['params'] = {}
    matches['params']['f_low'] = f_low
    matches['params']['f_match'] = f_match
    matches['params']['e'] = e
    matches['params']['M'] = chirp2total(chirp, q)
    matches['params']['q'] = q
    matches['params']['n'] = n
    matches['params']['n_gen'] = n_gen
    matches['params']['sample_rate'] = sample_rate
    matches['params']['approximant'] = approximant
    matches['params']['base_SNR'] = base_SNR
    matches['params']['distance'] = distance

    # Other keys
    matches['param_vals'] = param_vals
    matches['vary_param'] = vary_param

    return matches

def gen_grid_data(param_vals, base_vals,  MA_vals, n, n_gen, f_low, f_match, base_SNR, approximant='TEOBResumS'):

    all_matches = {}

    # Calculate harmonic ordering
    harm_ids = [0,1]
    for i in range(2,n):
        if harm_ids[-1] > 0:
            harm_ids.append(-harm_ids[-1])
        else:
            harm_ids.append(-harm_ids[-1]+1)

    # Calculate distance corresponding to base_SNR
    s = gen_wf(f_low, base_vals[0], chirp2total(base_vals[1], base_vals[2]), base_vals[2], sample_rate, approximant=approximant)
    psd = gen_psd(s, f_low)
    s_SNR = sigma(s.real(), psd, low_frequency_cutoff=f_match, high_frequency_cutoff=psd.sample_frequencies[-1])
    distance = s_SNR/base_SNR

    # Calculate grid for varying eccentricity
    start = time.time()
    all_matches['vary_e'] = grid_data('e', param_vals[0], MA_vals, n, n_gen, *base_vals, f_low, f_match, harm_ids, base_SNR, distance, approximant=approximant)
    end = time.time()
    print(f'Varying eccentricity: Completed in {end-start} seconds.')

    # Calculate grid for varying chirp mass
    start = time.time()
    all_matches['vary_chirp'] = grid_data('chirp', param_vals[1], MA_vals, n, n_gen, *base_vals, f_low, f_match, harm_ids, base_SNR, distance, approximant=approximant)
    end = time.time()
    print(f'Varying chirp mass: Completed in {end-start} seconds.')

    # Calculate grid for varying mass ratio
    start = time.time()
    all_matches['vary_q'] = grid_data('q', param_vals[2], MA_vals, n, n_gen, *base_vals, f_low, f_match, harm_ids, base_SNR, distance, approximant=approximant)
    end = time.time()
    print(f'Varying mass ratio: Completed in {end-start} seconds.')

    # Save all grids
    with open('all_matches', 'wb') as fp:
        pickle.dump(all_matches, fp)

if __name__ == "__main__":

    sample_rate = 4096

    # Generate and save grid data to desired data slot
    gen_grid_data([np.linspace(0, 0.4, 101), np.linspace(10, 40, 101), np.linspace(1, 5, 101)], [0.2, 24, 2], np.linspace(0, 2.5*np.pi, 40, endpoint=False), 4, 6, 10, 20, 20, approximant='TEOBResumS')
