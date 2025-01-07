#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import pickle
import os
from functools import partial
from pycbc.filter import match
import p_tqdm
from calcwf import *

def match_harms(wf_h, wf_t, f_low, psd, f_match=20, dominant_mode=0):

    # Perform match dominant mode
    wf_len = 2*len(wf_h[dominant_mode])
    for mode in wf_h.keys():
        wf_t[mode].resize(wf_len)
        wf_h[mode].resize(wf_len)

    m0_amp, m0_index, m0_phase = match(wf_t[dominant_mode].real(), wf_h[dominant_mode].real(), psd=psd,
                                       low_frequency_cutoff=f_match, subsample_interpolation=True, return_phase=True)

    matches = {}
    for k in wf_h.keys():
        h = wf_t[k].real()
        matches[k] = abs(overlap_cplx(wf_h[k].real(), h.cyclic_time_shift(m0_index * wf_h[dominant_mode].delta_t), psd=psd, low_frequency_cutoff=f_match))

    return matches

def single_match(e_chirp, wf_hjs_list, q, f_low, harm_ids, n_gen, psd, approximant='TEOBResumS'):

    # Unpack values and generate waveform
    e, chirp = e_chirp
    s = {}
    all_wfs = list(get_h([1]*n_gen, f_low, e, chirp2total(chirp, q), q, sample_rate, approximant=approximant))
    for i, id in enumerate(harm_ids):
        s[id] = all_wfs[i+1]

    # Calculate matches
    match_list = []
    for wf_hjs in wf_hjs_list:

        match_cplx = match_harms(wf_hjs, s, f_low, psd)
        match_list.append(list(match_cplx.values()))

    return match_list

def e_sqrd_chirp_grid_data(e_vals, chirp_vals, n, n_gen, fid_e_vals, fid_chirp, q, f_low, harm_ids, approximant='TEOBResumS'):

    # Generate fiducial waveform, resize to longest possible, and calculate psd
    wf_hjs_list = []
    long_wf = gen_wf(f_low, e_vals[0], chirp2total(chirp_vals[0], q), q, sample_rate, approximant)
    for fid_e in fid_e_vals:
        wf_hjs_list.append({})
        all_wfs = list(get_h([1]*n_gen, f_low, fid_e, chirp2total(fid_chirp, q), q, sample_rate, approximant=approximant))
        wf_hjs = resize_wfs(all_wfs[1:n+1], tlen=ceiltwo(len(long_wf)))
        for i, id in enumerate(harm_ids):
            wf_hjs_list[-1][id] = wf_hjs[i]
    h_psd = timeseries.TimeSeries(list(wf_hjs[0].copy())+[0], wf_hjs[0].delta_t, epoch=wf_hjs[0].start_time)
    psd = gen_psd(h_psd, f_low)

    # Generate list of all grid points
    e_vals_, chirp_vals_ = np.meshgrid(e_vals,chirp_vals,indexing='ij')
    e_chirp_vals = np.array([e_vals_.flatten(), chirp_vals_.flatten()]).T

    # Calculate all matches in parallel
    partial_single_match = partial(single_match, wf_hjs_list=wf_hjs_list, q=q, f_low=f_low, harm_ids=harm_ids, n_gen=n_gen, psd=psd, approximant=approximant)
    match_arr_list = np.array(p_tqdm.p_map(partial_single_match, e_chirp_vals))
    match_arr_list = np.swapaxes(match_arr_list, 0, 1)

    # Put match arrays into appropriate dictionary keys
    matches = {}
    for fid_e, match_arr in zip(fid_e_vals, match_arr_list):
        matches[fid_e] = {}
        for i, id in enumerate(harm_ids):
            matches[fid_e][f'h{id}'] = np.abs(match_arr[:,i].reshape(len(e_vals), len(chirp_vals)))
            matches[fid_e][f'h{harm_ids[i]}_phase'] = np.angle(match_arr[:,i].reshape(len(e_vals), len(chirp_vals)))

        # Add parameter keys
        matches[fid_e]['fid_params'] = {}
        matches[fid_e]['fid_params']['f_low'] = f_low
        matches[fid_e]['fid_params']['e'] = fid_e
        matches[fid_e]['fid_params']['M'] = chirp2total(fid_chirp, q)
        matches[fid_e]['fid_params']['q'] = q
        matches[fid_e]['fid_params']['n'] = n
        matches[fid_e]['fid_params']['sample_rate'] = sample_rate
        matches[fid_e]['fid_params']['approximant'] = approximant

        # Add grid size keys
        matches[fid_e]['e_vals'] = e_vals
        matches[fid_e]['chirp_vals'] = chirp_vals

    return matches

def gen_e_sqrd_chirp_data(fid_e_vals, fid_chirp_vals, e_vals, chirp_vals, n, n_gen, q, f_low, approximant='TEOBResumS'):

    all_matches = {}

    # Calculate harmonic ordering
    harm_ids = [0,1]
    for i in range(2,n):
        if harm_ids[-1] > 0:
            harm_ids.append(-harm_ids[-1])
        else:
            harm_ids.append(-harm_ids[-1]+1)

    # Calculate grid for each chirp mass at all fiducial eccentricities
    for i, fid_chirp in enumerate(fid_chirp_vals):
        start = time.time()
        all_matches[fid_chirp] = e_sqrd_chirp_grid_data(e_vals, chirp_vals[i], n, n_gen, fid_e_vals, fid_chirp, q, f_low, harm_ids, approximant=approximant)
        end = time.time()
        print(f'Chirp mass {fid_chirp}, eccentricities {fid_e_vals}: Completed in {end-start} seconds.')

    # Save all grids
    with open('all_matches', 'wb') as fp:
        pickle.dump(all_matches, fp)

if __name__ == "__main__":

    sample_rate = 4096

    # Generate and save grid data to desired data slot
    gen_e_sqrd_chirp_data([0.03519393, 0.07300364, 0.11175819], [10.06652832, 24.16193848, 40.27667236], np.linspace(0, 0.5, 151), [np.linspace(9.4,10.1,51), np.linspace(22,24.7,51), np.linspace(36.5,41,51)], 4, 20, 2, 10, approximant='TEOBResumS')
