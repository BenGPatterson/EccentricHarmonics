import math
import numpy as np
import matplotlib.pyplot as plt
import EOBRun_module
import astropy.constants as aconst
import scipy.constants as const
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from pycbc.waveform import td_approximants, fd_approximants, get_td_waveform, get_fd_waveform, taper_timeseries
from pycbc.detector import Detector
from pycbc.filter import match, optimized_match, overlap_cplx, sigma, sigmasq
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import timeseries, frequencyseries
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_total_mass, total_mass_and_mass_ratio_to_component_masses, component_masses_to_chirp_mass

## Conversions

def chirp2total(chirp, q):
    """
    Converts chirp mass to total mass.

    Parameters:
        chirp: Chirp mass.
        q: Mass ratio (m1/m2).

    Returns:
        Total mass.
    """
    return chirp_mass_and_mass_ratio_to_total_mass(chirp, 1/q)

def total2chirp(total, q):
    """
    Converts total mass to chirp mass.

    Parameters:
        total: Total mass.
        q: Mass ratio (m1/m2).

    Returns:
        Chirp mass.
    """
    
    return component_masses_to_chirp_mass(*total_mass_and_mass_ratio_to_component_masses(1/q, total))

def chirp_degeneracy_line(zero_ecc_chirp, ecc, sample_rate=4096, f_low=10, q=2, f_match=20, return_delta_m=False):
    """
    Calculates chirp masses corresponding to input eccentricities along a line of degeneracy 
    defined by a given chirp mass at zero eccentricity.

    Parameters:
        zero_ecc_chirp: Chirp mass of the degeneracy line at zero eccentricity.
        ecc: Eccentricities to find corresponding chirp masses for.
        sample_rate: Sampling rate to use when generating waveform.
        f_low: Starting frequency.
        q: Mass ratio.
        f_match: Low frequency cutoff to use.
        return_delta_m: Whether to also return delta m values.

    Returns:
        Chirp mass corresponding to each eccentricity.
    """
    
    # Generate waveform at non-eccentric point to use in sigmasq
    h = gen_wf(f_low, 0, chirp2total(zero_ecc_chirp, q), q, sample_rate, 'TEOBResumS')
    h.resize(ceiltwo(len(h)))

    # Generate the aLIGO ZDHP PSD
    psd = gen_psd(h, f_low)

    # Convert to frequency series
    h = h.real().to_frequencyseries()

    # Handle array of eccentricities as input
    array = False
    if len(np.shape(ecc)) > 0:
        array = True
    ecc = np.array(ecc).flatten()

    ssfs = np.zeros(len(ecc))
    ssffs = np.zeros(len(ecc))
    sskfs = np.zeros(len(ecc))
    sskffs = np.zeros(len(ecc))
    # Loop over each eccentricity
    for i, e in enumerate(ecc):
        
        # Calculate a few shifted es exactly
        sparse_s_fs = np.linspace(f_low, np.max([f_low*10,100]), 11)
        sparse_s_es = shifted_e(sparse_s_fs, f_low, e)
    
        # For low eccentricities use much faster approximate shifted e
        if sparse_s_fs[-1] < h.sample_frequencies[-1]:
            approx_s_fs = np.arange(sparse_s_fs[-1], h.sample_frequencies[-1], h.delta_f)+h.delta_f
            approx_s_es = shifted_e_approx(approx_s_fs, f_low, e)
            sparse_s_fs = np.concatenate([sparse_s_fs, approx_s_fs])
            sparse_s_es = np.concatenate([sparse_s_es, approx_s_es])
    
        # Interpolate to all frequencies
        s_e_interp = interp1d(sparse_s_fs, sparse_s_es, kind='cubic', fill_value='extrapolate')
        s_es = s_e_interp(h.sample_frequencies)
    
        # Calculate k values
        ks_sqrt = np.sqrt(2355*s_es**2/1462)
    
        # Calculate and normalise integrals
        ss = sigmasq(h, psd=psd, low_frequency_cutoff=f_match)
        ssf = sigmasq(h*h.sample_frequencies**(-5/6), psd=psd, low_frequency_cutoff=f_match)
        ssff = sigmasq(h*h.sample_frequencies**(-5/3), psd=psd, low_frequency_cutoff=f_match)
        sskf = -sigmasq(h*ks_sqrt*h.sample_frequencies**(-5/6), psd=psd, low_frequency_cutoff=f_match)
        sskff = -sigmasq(h*ks_sqrt*h.sample_frequencies**(-5/3), psd=psd, low_frequency_cutoff=f_match)
        ssfs[i], ssffs[i], sskfs[i], sskffs[i] = np.array([ssf, ssff, sskf, sskff])/ss

    # Calculate chirp mass
    delta_m = - (sskffs - ssfs*sskfs)/(ssffs - ssfs**2)
    chirp = zero_ecc_chirp*(1+delta_m)**(-3/5)

    # If array not passed then turn back into float
    if not array:
        chirp = chirp[0]
        delta_m = delta_m[0]

    if return_delta_m:
        return chirp, delta_m
    else:
        return chirp    

## Generating waveform

def modes_to_k(modes):
    """
    Converts list of modes to use into the 'k' parameter accepted by TEOBResumS.

    Parameters:
        modes: List of modes to use.

    Returns:
        'k' parameter of TEOBResumS.
    """
    
    return [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]

def gen_teob_wf(f, e, M, q, chi1, chi2, sample_rate, phase, distance, TA, inclination, mode_list):
    """
    Generates TEOBResumS waveform with chosen parameters.

    Parameters:
        f: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        chi1: Aligned spin of primary.
        chi2: Aligned spin of secondary.
        sample_rate: Sampling rate of waveform to be generated.
        phase: Phase of signal.
        distance: Luminosity distance to binary in Mpc.
        TA: Initial true anomaly.
        inclination: Inclination.
        mode_list: Modes to include.

    Returns:
        Plus and cross polarisation of TEOBResumS waveform.
    """

    # Define parameters
    k = modes_to_k(mode_list)
    pars = {
            'M'                  : M,
            'q'                  : q,    
            'chi1'               : chi1,
            'chi2'               : chi2,
            'domain'             : 0,            # TD
            'arg_out'            : 'no',         # Output hlm/hflm. Default = 0
            'use_mode_lm'        : k,            # List of modes to use/output through EOBRunPy
            'srate_interp'       : sample_rate,  # srate at which to interpolate. Default = 4096.
            'use_geometric_units': 'no',         # Output quantities in geometric units. Default = 1
            'initial_frequency'  : f,        # in Hz if use_geometric_units = 0, else in geometric units
            'interp_uniform_grid': 'yes',        # Interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
            'distance'           : distance,
            'coalescence_angle'  : phase,
            'inclination'        : 0,
            'ecc'                : e,
            'output_hpc'         : 'no',
            'ecc_freq'           : 3,
            'anomaly'            : TA,
            'inclination'        : inclination
            }

    # Calculate waveform and convert to pycbc TimeSeries object
    t, teob_p, teob_c = EOBRun_module.EOBRunPy(pars)
    teob = teob_p - 1j*teob_c
    tmrg = t[np.argmax(np.abs(teob))]
    t = t - tmrg
    teob_p = timeseries.TimeSeries(teob_p, 1/sample_rate, epoch=t[0])
    teob_c = timeseries.TimeSeries(teob_c, 1/sample_rate, epoch=t[0])
    
    return teob_p, teob_c

def gen_wf(f_low, e, M, q, sample_rate, approximant, chi1=0, chi2=0, phase=0, distance=1, TA=np.pi, inclination=0, mode_list=[[2,2]]):
    """
    Generates waveform with chosen parameters.

    Parameters:
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        sample_rate: Sampling rate of waveform to be generated.
        approximant: Approximant to use to generate the waveform.
        chi1: Aligned spin of primary.
        chi2: Aligned spin of secondary.
        phase: Phase of signal.
        distance: Luminosity distance to binary in Mpc.
        TA: Initial true anomaly.
        inclination: Inclination.
        mode_list: Modes to include.

    Returns:
        Complex combination of plus and cross waveform polarisations.
    """

    # Chooses specified approximant
    if approximant=='TEOBResumS':
        hp, hc = gen_teob_wf(f_low, e, M, q, chi1, chi2, sample_rate, phase, distance, TA, inclination, mode_list)
    else:
        raise Exception('approximant not recognised')

    # Returns waveform as complex timeseries
    return hp - 1j*hc

## Varying mean anomaly

def P_from_f(f):
    """
    Calculates orbital period from gravitational wave frequency.

    Parameters:
        f: Gravitational wave frequency.

    Returns:
        Orbital period.
    """
    
    f_orb = f/2
    return 1/f_orb

def a_from_P(P, M):
    """
    Calculates semi-major axis of orbit using Kepler's third law.

    Parameters:
        P: Orbital period.
        M: Total mass.

    Returns:
        Semi-major axis.
    """
    
    a_cubed = (const.G*M*P**2)/(4*np.pi**2)
    return a_cubed**(1/3)

def peri_advance_orbit(P, e, M):
    """
    Calculates periastron advance for one orbital revolution.

    Parameters:
        P: Orbital period.
        e: Eccentricity.
        M: Total mass.

    Returns:
        Periastron advance per orbit.
    """
    numerator = 6*np.pi*const.G*M
    a = a_from_P(P, M)
    denominator = const.c**2*a*(1-e**2)
    
    return numerator/denominator

def num_orbits(P, e, M):
    """
    Calculates number of orbits required for true anomaly to change by complete cycle of 2pi.

    Parameters:
        P: Orbital period.
        e: Eccentricity.
        M: Total mass.

    Returns:
        Number of orbits to shift true anomaly by 2pi.
    """
    
    delta_phi = peri_advance_orbit(P, e, M)
    n_orbit = (2*np.pi)/(2*np.pi - delta_phi)
    return n_orbit

def delta_freq_orbit(P, e, M, q):
    """
    Calculates shift in frequency for one orbital revolution.

    Parameters:
        P: Orbital period.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.

    Returns:
        Frequency shift per orbit.
    """
    
    m1, m2 = total_mass_and_mass_ratio_to_component_masses(1/q, M)
    numerator = 2*192*np.pi*(2*np.pi*const.G)**(5/3)*m1*m2*(1+(73/24)*e**2+(37/96)*e**4)
    denominator = 5*const.c**5*P**(8/3)*(m1+m2)**(1/3)*(1-e**2)**(7/2)
    return numerator/denominator

def shifted_f(f, e, M, q):
    """
    Calculates how to shift frequency such that anomaly changes by 2pi.

    Parameters:
        f: Original starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.

    Returns:
        Shifted starting frequency.
    """
    
    M *= aconst.M_sun.value
    P = P_from_f(f)
    delta_f_orbit = delta_freq_orbit(P, e, M, q)
    n_orbit = num_orbits(P, e, M)
    return f - delta_f_orbit*n_orbit

def shifted_e_approx(s_f, f, e):
    """
    Calculates how to shift eccentricity to match shifted frequency in such a way that the
    original frequency and eccentricity are recovered after one anomaly cycle of 2pi.
    Taylor expansion to lowest order in e.

    Parameters:
        s_f: Shifted starting frequency.
        f: Original starting frequency.
        e: Starting eccentricity.

    Returns:
        Shifted starting eccentricity.
    """  

    s_e = e*(s_f/f)**(-19/18)
    return s_e

def shifted_e_const(f, e):
    """
    Calculates constant of proportionality between gw frequency and function of eccentricity.

    Parameters:
        f: Gravitational wave frequency.
        e: Eccentricity.

    Returns:
        Proportionality constant.
    """

    constant = f*e**(18/19)*(1+(121/304)*e**2)**(1305/2299)*(1-e**2)**(-3/2)

    return constant

def shifted_e(s_f, f, e):
    """
    Calculates how to shift eccentricity to match shifted frequency in such a way that the
    original frequency and eccentricity are recovered after one anomaly cycle of 2pi.

    Parameters:
        s_f: Shifted starting frequency.
        f: Original starting frequency.
        e: Starting eccentricity.

    Returns:
        Shifted starting eccentricity.
    """ 

    # Ensure inputs are arrays
    array = False
    if len(np.shape(s_f))+len(np.shape(e)) > 0:
        array = True
    s_f = np.array(s_f).flatten()
    e = np.array(e).flatten()

    # Compute shifted eccentricity
    constant = shifted_e_const(f, e)
    bounds = [(0, 0.999)]
    s_e_approx = shifted_e_approx(s_f, f, e)
    init_guess = np.min([s_e_approx, np.full(len(s_e_approx), bounds[0][1])], axis=0)
    best_fit = minimize(lambda x: np.sum(abs(shifted_e_const(s_f, x)-constant)**2), init_guess, bounds=bounds)
    s_e = np.array(best_fit['x'])
    if not array:
        s_e = s_e[0]

    return s_e

## Match waveforms

def gen_psd(h_psd, f_low):
    """
    Generates psd required for a real time series.

    Parameters:
        h_psd: Time series to generate psd for.
        f_low: Starting frequency of waveform.

    Returns:
        Psd.
    """

    # Resize wf to next highest power of two
    h_psd.resize(ceiltwo(len(h_psd)))

    # Generate the aLIGO ZDHP PSD
    delta_f = 1.0 / h_psd.duration
    flen = len(h_psd)//2 + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_low)

    return psd

def ceiltwo(number):
    """
    Finds next highest power of two of a number.

    Parameters:
        number: Number to find next highest power of two for.

    Returns:
        Next highest power of two.
    """
    
    ceil = math.ceil(np.log2(number))
    return 2**ceil

def resize_wfs(wfs, tlen=None):
    """
    Resizes two or more input waveforms to all match the next highest power of two.

    Parameters:
        wfs: List of input waveforms.
        tlen: Length to resize to.

    Returns:
        Resized waveforms.
    """

    if tlen is None:
        lengths = [len(i) for i in wfs]
        tlen = ceiltwo(max(lengths))
    for wf in wfs:
        wf.resize(tlen)
    return wfs

def trim_wf(wf_trim, wf_ref):
    """
    Cuts the initial part of one of the waveforms such that both have the same amount of data prior to merger.

    Parameters:
        wf_trim: Waveform to be edited.
        wf_ref: Reference waveform.
        
    Returns:
        Edited waveform.
    """

    wf_trim_interpolate = interp1d(wf_trim.sample_times, wf_trim, bounds_error=False, fill_value=0)
    wf_trim_strain = wf_trim_interpolate(wf_ref.sample_times)
    wf_trim = timeseries.TimeSeries(wf_trim_strain, wf_ref.delta_t, epoch=wf_ref.start_time)
    assert np.array_equal(wf_ref.sample_times, wf_trim.sample_times)

    return wf_trim

def prepend_zeros(wf_pre, wf_ref):
    """
    Prepends zeros to one of the waveforms such that both have the same amount of data prior to merger.

    Parameters:
        wf_pre: Waveform to be edited.
        wf_ref: Reference waveform.
        
    Returns:
        Edited waveform.
    """

    wf_pre_interpolate = interp1d(wf_pre.sample_times, wf_pre, bounds_error=False, fill_value=0)
    wf_pre_strain = wf_pre_interpolate(wf_ref.sample_times)
    wf_pre = timeseries.TimeSeries(wf_pre_strain, wf_ref.delta_t, epoch=wf_ref.start_time)
    assert np.array_equal(wf_ref.sample_times, wf_pre.sample_times)

    return wf_pre

def match_hn(wf_hjs_, wf_s, f_low, f_match=20, return_index=False, psd=None):
    """
    Calculates match between dominant waveform and a trial waveform, and uses the time shift 
    in this match to compute the complex overlaps between the time-shifted sub-dominant waveforms
    and the trial waveform. This ensures the 'match' is calculated for all harmonics at the same 
    time.

    Parameters:
        wf_hjs_: List of harmonic waveforms.
        wf_s: Trial waveform.
        f_low: Starting frequency of waveforms.
        f_match: Low frequency cutoff to use. 
        return_index: Whether to return index shift of dominant harmonic match.
        psd: psd to use.
        
    Returns:
        Complex matches of trial waveform to harmonics.
    """

    # Creates new versions of waveforms to avoid editing originals
    wf_hjs = []
    for i in range(len(wf_hjs_)):
        wf_new = timeseries.TimeSeries(wf_hjs_[i].copy(), wf_hjs_[i].delta_t, epoch=wf_hjs_[i].start_time)
        wf_hjs.append(wf_new)
    wf_s = timeseries.TimeSeries(wf_s.copy(), wf_s.delta_t, epoch=wf_s.start_time)

    # Generate the aLIGO ZDHP PSD
    if psd is None:
        if len(wf_hjs[0]) > len(wf_s):
            psd = gen_psd(wf_hjs[0], f_low)
        else:
            psd = gen_psd(wf_s, f_low)

    # Resize waveforms to the length of the psd
    tlen = (len(psd)-1)*2
    all_wfs = resize_wfs([*wf_hjs, wf_s], tlen=tlen)
    wf_hjs = all_wfs[:-1]
    wf_s = all_wfs[-1]
    wf_len = len(wf_s)

    # Perform match on dominant
    m_h1_amp, m_index, m_h1_phase = match(wf_hjs[0].real(), wf_s.real(), psd=psd, low_frequency_cutoff=f_match, subsample_interpolation=True, return_phase=True)
    m_h1 = m_h1_amp*np.e**(1j*m_h1_phase)

    # Shift sub-dominant
    if m_index <= len(wf_hjs[0])/2:
        # If sub-dominant needs to be shifted forward, prepend zeros to it
        for i in range(1,len(wf_hjs)):
            wf_hjs[i].prepend_zeros(int(m_index))
            wf_hjs[i].resize(wf_len)
    else:
        # If sub-dominant needs to be shifted backward, prepend zeros to trial waveform instead
        wf_s.prepend_zeros(int(len(wf_hjs[0]) - m_index))
        wf_s.resize(wf_len)

    # As subsample_interpolation=True, require interpolation of sub-dominant to account for non-integer index shift
    delta_t = wf_hjs[0].delta_t
    if m_index <= len(wf_hjs[0])/2:
        # If sub-dominant needs to be shifted forward, interpolate sub-dominant forward
        inter_index = m_index - int(m_index)
        for i in range(1,len(wf_hjs)):
            wf_hj_interpolate = interp1d(wf_hjs[i].sample_times, wf_hjs[i], bounds_error=False, fill_value=0)
            wf_hj_strain = wf_hj_interpolate(wf_hjs[i].sample_times-(inter_index*delta_t))
            wf_hjs[i] = timeseries.TimeSeries(wf_hj_strain, wf_hjs[i].delta_t, epoch=wf_hjs[i].start_time-(inter_index*delta_t))
    else:
        # If sub-dominant needs to be shifted backward, interpolate sub-dominant backward
        inter_index = (len(wf_hjs[0]) - m_index) - int(len(wf_hjs[0]) - m_index)
        for i in range(1,len(wf_hjs)):
            wf_hj_interpolate = interp1d(wf_hjs[i].sample_times, wf_hjs[i], bounds_error=False, fill_value=0)
            wf_hj_strain = wf_hj_interpolate(wf_hjs[i].sample_times+(inter_index*delta_t))
            wf_hjs[i] = timeseries.TimeSeries(wf_hj_strain, wf_hjs[i].delta_t, epoch=wf_hjs[i].start_time+(inter_index*delta_t))

    # Perform complex overlap on sub-dominant
    matches = [m_h1]
    for i in range(1,len(wf_hjs)):
        m = overlap_cplx(wf_hjs[i].real(), wf_s.real(), psd=psd, low_frequency_cutoff=f_match)
        matches.append(m)
    
    # Returns index shift if requested
    if return_index:
        return *matches, m_index
    else:
        return matches    

def match_wfs(wf1, wf2, f_low, subsample_interpolation, f_match=20, return_phase=False):
    """
    Calculates match (overlap maximised over time and phase) between two input waveforms.

    Parameters:
        wf1: First input waveform.
        wf2: Second input waveform.
        f_low: Lower bound of frequency integral.
        subsample_interpolation: Whether to use subsample interpolation.
        f_match: Low frequency cutoff to use.
        return_phase: Whether to return phase of maximum match.
        
    Returns:
        Amplitude (and optionally phase) of match.
    """

    # Resize the waveforms to the same length
    wf1, wf2 = resize_wfs([wf1, wf2])

    # Generate the aLIGO ZDHP PSD
    psd = gen_psd(wf1, f_low)

    # Perform match
    m = match(wf1.real(), wf2.real(), psd=psd, low_frequency_cutoff=f_match, subsample_interpolation=subsample_interpolation, return_phase=return_phase)

    # Additionally returns phase required to match waveforms up if requested
    if return_phase:
        return m[0], m[2]
    else:
        return m[0]

def overlap_cplx_wfs(wf1, wf2, f_low, f_match=20, normalized=True):
    """
    Calculates complex overlap (overlap maximised over phase) between two input waveforms.

    Parameters:
        wf1: First input waveform.
        wf2: Second input waveform.
        f_low: Starting frequency of waveforms.
        f_match: Low frequency cutoff to use.
        normalized: Whether to normalise result between 0 and 1.
        
    Returns:
        Complex overlap.
    """

    # Prepends earlier wf with zeroes so same amount of data before merger (required for overlap_cplx)
    if wf1.start_time > wf2.start_time:
        wf1 = prepend_zeros(wf1, wf2)
    elif wf1.start_time < wf2.start_time:
        wf2 = prepend_zeros(wf2, wf1)
    assert wf1.start_time == wf2.start_time

    # Ensures wfs are tapered
    if wf1[0] != 0:
        wf1 = taper_wf(wf1)
    if wf2[0] != 0:
        wf2 = taper_wf(wf2)
    
    # Resize the waveforms to the same length
    wf1, wf2 = resize_wfs([wf1, wf2])

    # Generate the aLIGO ZDHP PSD
    psd = gen_psd(wf1, f_low)

    # Perform complex overlap
    m = overlap_cplx(wf1.real(), wf2.real(), psd=psd, low_frequency_cutoff=f_match, normalized=normalized)

    return m

## Waveform components

def taper_wf(wf_taper):
    """
    Tapers start of input waveform using pycbc.waveform taper_timeseries() function.

    Parameters:
        wf_taper: Waveform to be tapered.
        
    Returns:
        Tapered waveform.
    """
    
    wf_taper_p = taper_timeseries(wf_taper.real(), tapermethod='start')
    wf_taper_c = taper_timeseries(-wf_taper.imag(), tapermethod='start')
    wf_taper = wf_taper_p - 1j*wf_taper_c

    return wf_taper

def get_comp_shifts(f_low, e, M, q, n):
    '''
    Calculates shifted frequency and eccentricity required to create each component
    waveform (beyond first).

    Parameters:
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        n: Number of waveform components.

    Returns:
        Shifted frequency and eccentricity for all components beyond first.
    '''

    # Finds shifted frequency and eccentricity without correction
    max_s_f = shifted_f(f_low, e, M, q)
    s_f_vals = np.linspace(f_low, max_s_f, n, endpoint=False)[1:]
    s_e_vals = shifted_e(s_f_vals, f_low, e)

    return s_f_vals, s_e_vals

def gen_component_wfs(f_low, e, M, q, n, sample_rate, approximant, normalisation, phase, f_match):
    '''
    Creates n component waveforms used to make harmonics, all equally spaced in
    mean anomaly at a fixed time before merger.
    
    Parameters:
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        n: Number of waveform components.
        sample_rate: Sample rate of waveform.
        approximant: Approximant to use.
        normalisation: Whether to normalise x_0,...,x_n-1 components to ensure (x_j|x_j) is constant.
        phase: Initial phase of x_0,...,x_n-1 components.
        f_match: Low frequency cutoff to use.
        
    Returns:
        Component waveforms.
    '''

    # Generates first (unshifted) component waveform and shifts required for others
    h = gen_wf(f_low, e, M, q, sample_rate, approximant, phase=phase)
    s_f_vals, s_e_vals = get_comp_shifts(f_low, e, M, q, n)

    # Tapers first waveform
    h = taper_wf(h)
    
    # Calculates normalisation factor using sigma function
    if normalisation:
        # Generate the aLIGO ZDHP PSD
        h.resize(ceiltwo(len(h))) 
        psd = gen_psd(h, f_low)
        sigma_0 = sigma(h.real(), psd=psd, low_frequency_cutoff=f_match)

    comp_wfs = [h]
    
    # Generate all component waveforms
    for i in range(n-1):

        # Create waveform
        h = gen_wf(s_f_vals[i], s_e_vals[i], M, q, sample_rate, approximant, phase=phase)

        # Trim waveform to same size as first (shortest), and corrects phase
        h = trim_wf(h, comp_wfs[0])
        overlap = overlap_cplx_wfs(h, comp_wfs[0], f_low, f_match=f_match)
        phase_angle = -np.angle(overlap)/2
        h *= np.exp(2*1j*phase_angle)
        h = trim_wf(h, comp_wfs[0])

        # Tapers
        h = taper_wf(h)
        
        # Normalises waveform if requested
        if normalisation:
            sigma_h = sigma(h.real(), psd=psd, low_frequency_cutoff=f_match)
            h *= sigma_0/sigma_h

        comp_wfs.append(h)

    return comp_wfs

def get_dominance_order(n):
    '''
    Creates indexing array to order waveforms from their natural roots of unity order 
    to their order of dominance: h0, h1, h-1, h2, h3, h4, ...
    
    Parameters:
        n: Number of waveform components.
        
    Returns:
        Indexing array.
    '''

    # Start with roots of unity ordering
    j_order = list(np.arange(n))

    # Move -1 harmonic if required
    if n>= 4:
        j_order.insert(2, j_order[-1])
        j_order = j_order[:-1]

    return j_order

def GS_proj(u, v, f_low, f_match, psd):
    '''
    Performs projection used in Grant-Schmidt orthogonalisation, defined as 
    u*(v|u)/(u|u).
    
    Parameters:
        u: Waveform u defined above.
        v: Waveform v defined above.
        f_low: Starting frequency.
        f_match: Low frequency cutoff to use.
        psd: Psd to use to weight complex overlap.
        
    Returns:
        Projection u*(v|u)/(u|u).
    '''

    numerator = overlap_cplx(v.real(), u.real(), psd=psd, low_frequency_cutoff=f_match, normalized=False)
    denominator = overlap_cplx(u.real(), u.real(), psd=psd, low_frequency_cutoff=f_match, normalized=False)

    return u*numerator/denominator

def GS_orthogonalise(f_low, f_match, wfs):
    '''
    Performs Grant-Schmidt orthogonalisation on harmonic waveforms to ensure 
    (hj|hm) = 0 for j!=m.
    
    Parameters:
        f_low: Starting frequency.
        f_match: Low frequency cutoff to use.
        wfs: Harmonic waveforms.
        
    Returns:
        Grant-Schmidt orthogonalised harmonics.
    '''

    # Generates psd for use in orthogonalisation
    psd = gen_psd(wfs[0], f_low)

    # Orthogonalises each waveform in turn
    for i in range(1,len(wfs)):
        for j in range(i):
            wfs[i] = wfs[i] - GS_proj(wfs[j], wfs[i], f_low, f_match, psd)

    return wfs

def get_ortho_ovlps(h_wfs, f_low, f_match=20):
    """
    Calculate overlaps between unorthogonalised set of harmonics, and 
    compute the overlap of orthogonalised harmonics with themselves.

    Parameters:
        h_wfs: Unorthogonalised harmonics.
        f_low: Starting frequency.
        f_match: Low frequency cutoff to use.

    Returns:
        ovlps: Overlaps of unorthogonalised harmonics.
        ovlps_perp: Overlaps of orthogonalised harmonics with themselves.
    """

    # Calculate psd
    psd = gen_psd(h_wfs[0], f_low)

    # Normalise wfs
    for i in range(len(h_wfs)):
        h_wf_f = h_wfs[i].real().to_frequencyseries()
        h_wfs[i] /= sigma(h_wf_f, psd, low_frequency_cutoff=f_match, high_frequency_cutoff=psd.sample_frequencies[-1])

    # Calculate all overlap combinations
    n = len(h_wfs)
    ovlps = {}
    for i in range(1,n):
        ovlps[i] = {}
        for j in range(i):
            ovlps[i][j] = overlap_cplx(h_wfs[i].real(), h_wfs[j].real(), psd=psd, low_frequency_cutoff=f_match, normalized=False)
            
    # Compute orthogonal overlaps
    ovlps_perp = {}
    for i in range(n):
        abs_sqrd = 0
        for j in range(i):
            abs_sqrd += np.abs(ovlps[i][j])**2
        triple_ovlps = 0
        for j in range(i):
            for k in range(j):
                triple_ovlps += ovlps[i][j]*np.conj(ovlps[i][k])*ovlps[j][k]
        ovlps_perp[i] = 1 - abs_sqrd + 2*np.real(triple_ovlps)

    return ovlps, ovlps_perp

def get_h_TD(f_low, coeffs, comp_wfs, GS_normalisation, f_match, return_ovlps=False):
    """
    Combines waveform components in time domain to form harmonics and total h as follows:

    Parameters:
        f_low: Starting frequency.
        coeffs: List containing coefficients of harmonics.
        comp_wfs: Waveform components x_0, ..., x_n-1.
        GS_normalisation: Whether to perform Grant-Schmidt orthogonalisaton to ensure (hj|hm) = 0 for j!=m.
        f_match: Low frequency cutoff to use.
        return_ovlps: Whether to return overlaps between all unorthogonalised harmonics.
        
    Returns:
        All waveform components and combinations: total, *harmonics, *components
    """

    # Find first primitive root of unity
    prim_root = np.e**(2j*np.pi/len(coeffs))
    
    # Build harmonics
    hs = []
    for i in range(len(coeffs)):
        hs.append((1/len(coeffs))*comp_wfs[0])
        for j in range(len(coeffs)-1):
            hs[-1] += (1/len(coeffs))*comp_wfs[j+1]*prim_root**(i*(j+1))

    # Re-order by dominance rather than natural roots of unity order
    j_order = get_dominance_order(len(coeffs))
    hs = [hs[i] for i in j_order]

    # Calculates overlaps if requested
    ovlps, ovlps_perp = None, None
    if return_ovlps:
        ovlps, ovlps_perp = get_ortho_ovlps(hs, f_low, f_match=f_match)

    # Perform Grant-Schmidt orthogonalisation if requested
    if GS_normalisation:
        hs = GS_orthogonalise(f_low, f_match, hs)

    # Calculates overall waveform using complex coefficients A, B, C, ...
    h = coeffs[0]*hs[0]
    for i in range(len(coeffs)-1):
        h += coeffs[i+1]*hs[i+1]
    
    # Returns overall waveform and components for testing purposes
    return [h, *hs, *comp_wfs], ovlps, ovlps_perp

def get_h(coeffs, f_low, e, M, q, sample_rate, approximant='TEOBResumS', f_match=20, subsample_interpolation=True, GS_normalisation=True, comp_normalisation=False, comp_phase=0, return_ovlps=False):
    """
    Generates a overall h waveform, harmonic waveforms, and component waveforms.

    Parameters:
        coeffs: List containing coefficients of harmonics.
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        sample_rate: Sample rate of waveform.
        approximant: Approximant to use.
        f_match: Low frequency cutoff to use.
        subsample_interpolation: Whether to use subsample interpolation.
        GS_normalisation: Whether to perform Grant-Schmidt orthogonalisaton to ensure (hj|hm) = 0 for j!=m.
        comp_normalisation: Whether to normalise x_0,...,x_n-1 components to ensure (sj|sj) is constant.
        comp_phase: Initial phase of x_0,...,x_n-1 components.
        return_ovlps: Whether to return overlaps between all unorthogonalised harmonics.
        
    Returns:
        All waveform components and combinations: total, *harmonics, *components
    """

    # Other approximants are deprecated
    assert approximant == 'TEOBResumS'

    # Gets (normalised) components which make up overall waveform
    component_wfs = gen_component_wfs(f_low, e, M, q, len(coeffs), sample_rate, approximant, comp_normalisation, comp_phase, f_match)

    # Calculate overall waveform and components in time domain
    wfs, ovlps, ovlps_perp = get_h_TD(f_low, coeffs, component_wfs, GS_normalisation, f_match, return_ovlps=return_ovlps)

    if return_ovlps:
        return wfs, ovlps, ovlps_perp
    else:    
        return wfs