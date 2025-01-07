#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import pickle
import os
from functools import partial
import p_tqdm
from calcwf import *
from interpolating_match import *

def single_match(s_f_e, e_chirp, wf_hjs, q, f_low, ovlps, ovlps_perp, harm_ids, psd, approximant='TEOBResumS'):

    # Unpack values and generate waveform
    s_f, s_e = s_f_e
    e, chirp = e_chirp
    s = gen_wf(s_f, s_e, chirp2total(chirp, q), q, sample_rate, approximant=approximant)

    # Calculate matches
    match_cplx = match_hn(wf_hjs, s, f_low, psd=psd)

    # Get match quantities
    match_quantities = []
    for i in range(1, len(wf_hjs)):
        match_quantities.append(np.abs(match_cplx[i])/np.abs(match_cplx[0])) # Single harmonic
        if i > 1:
            num = 0
            for j in range(1,i+1):
                num += np.abs(match_cplx[j])**2
            match_quantities.append(np.sqrt(num/np.abs(match_cplx[0]**2))) # Naive multiple harmonics
            if i == 2:
                pc_frac = comb_harm_consistent(np.abs(match_cplx[:i+1]), np.angle(match_cplx[:i+1]), harms=harm_ids[:i+1])
                match_quantities.append(pc_frac) # Phase consistent multiple harmonics

    # Save memory
    del s

    return *match_cplx, np.linalg.norm(match_cplx), *match_quantities

def chirp_match_MA_grid_data(param_vals, MA_vals, n, n_gen, fid_e, zero_ecc_chirp, q, f_low, harm_ids, approximant='TEOBResumS'):

    # Generate param values along line of degeneracy
    all_e_vals = np.array([fid_e, *param_vals]).flatten()
    all_chirp_vals = chirp_degeneracy_line(zero_ecc_chirp, all_e_vals, sample_rate, f_low=f_low, q=q)
    fid_e, e_vals = all_e_vals[0], all_e_vals[1:]
    fid_chirp, chirp_vals = all_chirp_vals[0], all_chirp_vals[1:]
    e_chirp_vals = list(map(list, zip(e_vals, chirp_vals)))
    e_chirp_vals = list(np.repeat(e_chirp_vals, len(MA_vals), axis=0))

    # Generate fiducial waveform
    all_wfs, ovlps, ovlps_perp = get_h([1]*n_gen, f_low, fid_e, chirp2total(fid_chirp, q), q, sample_rate, approximant=approximant, return_ovlps=True)
    wf_hjs = all_wfs[1:n+1]
    del all_wfs

    s_f_e_vals = []
    # Loop over chirp mass values
    for e, chirp in zip(e_vals, chirp_vals):

        # Find shifted_e, shifted_f for all MA values
        s_f_2pi = f_low - shifted_f(f_low, e, chirp2total(chirp, q), q)
        s_f_vals = f_low - MA_vals*s_f_2pi/(2*np.pi)
        s_e_vals = shifted_e(s_f_vals, f_low, e)
        s_f_e_vals += list(map(list, zip(s_f_vals, s_e_vals)))

    # Resize fiducial waveforms to longest possible waveform, and calculate psd
    long_wf = gen_wf(s_f_e_vals[len(MA_vals)-1][0], s_f_e_vals[len(MA_vals)-1][1], chirp2total(chirp_vals[0], q), q, sample_rate, approximant)
    wf_hjs = resize_wfs(wf_hjs, tlen=ceiltwo(len(long_wf)))
    h_psd = timeseries.TimeSeries(list(wf_hjs[0].copy())+[0], wf_hjs[0].delta_t, epoch=wf_hjs[0].start_time)
    psd = gen_psd(h_psd, f_low)

    # Calculate all matches in parallel
    partial_single_match = partial(single_match, wf_hjs=wf_hjs, q=q, f_low=f_low, ovlps=ovlps, ovlps_perp=ovlps_perp, harm_ids=harm_ids, psd=psd, approximant=approximant)
    match_arr = np.array(p_tqdm.p_map(partial_single_match, s_f_e_vals, e_chirp_vals))

    # Save memory
    del wf_hjs

    # Put match arrays into appropriate dictionary keys
    matches = {}
    for i in range(n):
        matches[f'h{harm_ids[i]}'] = np.abs(match_arr[:,i].reshape(-1, len(MA_vals)))
        matches[f'h{harm_ids[i]}_phase'] = np.angle(match_arr[:,i].reshape(-1, len(MA_vals)))
    matches['quad'] = match_arr[:,n].reshape(-1, len(MA_vals))
    count = n+1
    for i in range(1,n):
        matches[f'h{harm_ids[i]}_h0'] = match_arr[:,count].reshape(-1, len(MA_vals))
        count += 1
        if i > 1:
            str_combo = ''
            for j in range(1, i+1):
                str_combo += f'h{harm_ids[j]}_'
            matches[f'{str_combo}h0'] = match_arr[:,count].reshape(-1, len(MA_vals))
            count += 1
            if i == 2:
                matches[f'{str_combo}h0_pc'] = match_arr[:,count].reshape(-1, len(MA_vals))
                count += 1

    # Add parameter keys
    matches['fid_params'] = {}
    matches['fid_params']['f_low'] = f_low
    matches['fid_params']['e'] = fid_e
    matches['fid_params']['M'] = chirp2total(fid_chirp, q)
    matches['fid_params']['q'] = q
    matches['fid_params']['n'] = n
    matches['fid_params']['n_gen'] = n_gen
    matches['fid_params']['sample_rate'] = sample_rate
    matches['fid_params']['approximant'] = approximant

    # Other keys
    matches['e_vals'] = e_vals
    matches['ovlps'] = ovlps
    matches['ovlps_perp'] = ovlps_perp

    return matches

def gen_grid_data(scaling_norms, e_vals, MA_vals, n, n_gen, fid_e_vals, q, f_low, approximant='TEOBResumS'):

    all_matches = {}

    # Calculate harmonic ordering
    harm_ids = [0,1]
    for i in range(2,n):
        if harm_ids[-1] > 0:
            harm_ids.append(-harm_ids[-1])
        else:
            harm_ids.append(-harm_ids[-1]+1)

    # Calculate grid for each chirp mass
    for fid_e in fid_e_vals:
        start = time.time()
        zero_ecc_chirp = fid_e**(6/5)*scaling_norms[0]/(scaling_norms[1]**(6/5))
        all_matches[zero_ecc_chirp] = chirp_match_MA_grid_data(e_vals, MA_vals, n, n_gen, fid_e, zero_ecc_chirp, q, f_low, harm_ids, approximant=approximant)
        end = time.time()
        print(f'Non-eccentric chirp mass: {zero_ecc_chirp}, fiducial eccentricity: {fid_e}: Completed in {end-start} seconds.')

    # Save all grids
    with open('all_matches', 'wb') as fp:
        pickle.dump(all_matches, fp)

if __name__ == "__main__":

    sample_rate = 4096

    # Generate and save grid data to desired data slot
    gen_grid_data([10, 0.035], np.linspace(0, 0.5, 201), np.linspace(0, 2.5*np.pi, 40, endpoint=False), 4, 6, [0.03519393, 0.07300364, 0.11175819], 2, 10, approximant='TEOBResumS')
