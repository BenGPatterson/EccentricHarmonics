import EOBRun_module
import numpy as np
import math
import scipy.constants as const
import astropy.constants as aconst
from pycbc.waveform import get_td_waveform, taper_timeseries
from pycbc.types import timeseries
from pycbc.filter import match, overlap_cplx, sigma
from pycbc.psd import aLIGOZeroDetHighPower
from scipy.optimize import minimize
from scipy.interpolate import interp1d

## Conversions

def f_kep2avg(f_kep, e):
    """
    Converts Keplerian frequency to the average frequency quantity used by TEOBResumS.

    Parameters:
        f_kep: Keplerian frequency to be converted.
        e: eccentricity of signal.

    Returns:
        Average frequency.
    """

    numerator = (1+e**2)
    denominator = (1-e**2)**(3/2)

    return f_kep*(numerator/denominator)

def chirp2total(chirp, q):
    """
    Converts chirp mass to total mass.

    Parameters:
        chirp: Chirp mass.
        q: Mass ratio.

    Returns:
        Total mass.
    """
    
    q_factor = q/(1+q)**2
    total = q_factor**(-3/5) * chirp

    return total

## Generating waveform

def gen_e_td_wf(f_low, e, M, q, sample_rate, phase, distance):
    """
    Generates EccentricTD waveform with chosen parameters.

    Parameters:
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        sample_rate: Sampling rate of waveform to be generated.
        phase: Phase of signal.
        distance: Luminosity distance to binary in Mpc.

    Returns:
        Plus and cross polarisation of EccentricTD waveform.
    """
    
    m2 = M / (1+q)
    m1 = M - m2
    e_td_p, e_td_c = get_td_waveform(approximant='EccentricTD',
                                     mass1=m1,
                                     mass2=m2,
                                     eccentricity=e,
                                     coa_phase=phase,
                                     distance=distance,
                                     delta_t=1.0/sample_rate,
                                     f_lower=f_low)
    return e_td_p, e_td_c

def modes_to_k(modes):
    """
    Converts list of modes to use into the 'k' parameter accepted by TEOBResumS.

    Parameters:
        modes: List of modes to use.

    Returns:
        'k' parameter of TEOBResumS.
    """
    
    return [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]

def gen_teob_wf(f_kep, e, M, q, sample_rate, phase, distance):
    """
    Generates TEOBResumS waveform with chosen parameters.

    Parameters:
        f_kep: Starting (Keplerian) frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        sample_rate: Sampling rate of waveform to be generated.
        phase: Phase of signal.
        distance: Luminosity distance to binary in Mpc.

    Returns:
        Plus and cross polarisation of TEOBResumS waveform.
    """

    # Gets average frequency quantity used by TEOBResumS
    f_avg = f_kep2avg(f_kep, e)

    # Define parameters
    k = modes_to_k([[2,2]])
    pars = {
            'M'                  : M,
            'q'                  : q,
            'Lambda1'            : 0.,
            'Lambda2'            : 0.,     
            'chi1'               : 0.,
            'chi2'               : 0.,
            'domain'             : 0,            # TD
            'arg_out'            : 0,            # Output hlm/hflm. Default = 0
            'use_mode_lm'        : k,            # List of modes to use/output through EOBRunPy
            'srate_interp'       : sample_rate,  # srate at which to interpolate. Default = 4096.
            'use_geometric_units': 0,            # Output quantities in geometric units. Default = 1
            'initial_frequency'  : f_avg,        # in Hz if use_geometric_units = 0, else in geometric units
            'interp_uniform_grid': 1,            # Interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
            'distance'           : distance,
            'coalescence_angle'  : phase,
            'inclination'        : 0,
            'ecc'                : e,
            'output_hpc'         : 0
            }

    # Calculate waveform and convert to pycbc TimeSeries object
    t, teob_p, teob_c = EOBRun_module.EOBRunPy(pars)
    teob = teob_p - 1j*teob_c
    tmrg = t[np.argmax(np.abs(teob))]
    t = t - tmrg
    teob_p = timeseries.TimeSeries(teob_p, 1/sample_rate, epoch=t[0])
    teob_c = timeseries.TimeSeries(teob_c, 1/sample_rate, epoch=t[0])
    
    return teob_p, teob_c

def gen_wf(f_low, e, M, q, sample_rate, approximant, phase=0, distance=1):
    """
    Generates waveform with chosen parameters.

    Parameters:
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        sample_rate: Sampling rate of waveform to be generated.
        approximant: Approximant to use to generate the waveform.
        phase: Phase of signal.
        distance: Luminosity distance to binary in Mpc.

    Returns:
        Complex combination of plus and cross waveform polarisations.
    """

    # Chooses specified approximant
    if approximant=='EccentricTD':
        hp, hc = gen_e_td_wf(f_low, e, M, q, sample_rate, phase, distance)
    elif approximant=='TEOBResumS':
        hp, hc = gen_teob_wf(f_low, e, M, q, sample_rate, phase, distance)
    else:
        raise Exception('approximant not recognised')

    # Returns waveform as complex timeseries
    return hp - 1j*hc

## Varying mean anomaly

def m1_m2_from_M_q(M, q):
    """
    Calculates component masses from total mass and mass ratio.

    Parameters:
        M: Total mass.
        q: Mass ratio.

    Returns:
        Masses of binary components.
    """
    
    m2 = M/(1+q)
    m1 = M - m2
    return m1, m2

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
    
    m1, m2 = m1_m2_from_M_q(M, q)
    numerator = 2*192*np.pi*(2*np.pi*const.G)**(5/3)*m1*m2*(1+(73/24)*e**2+(37/96)*e**4)
    denominator = 5*const.c**5*P**(8/3)*(m1+m2)**(1/3)*(1-e**2)**(7/2)
    return numerator/denominator

def shifted_f(f, e, M, q):
    """
    Calculates how to shift frequency such that true anomaly changes by 2pi.

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
    original frequency and eccentricity are recovered after one true anomaly cycle of 2pi.
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
    original frequency and eccentricity are recovered after one true anomaly cycle of 2pi.

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
    best_fit = minimize(lambda x: np.sum(abs(shifted_e_const(s_f, x)-constant)), init_guess, bounds=bounds)
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

def resize_wfs(wfs):
    """
    Resizes two or more input waveforms to all match the next highest power of two.

    Parameters:
        wfs: List of input waveforms.

    Returns:
        Resized waveforms.
    """
    
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

def match_wfs(wf1, wf2, f_low, subsample_interpolation, return_phase=False):
    """
    Calculates match (overlap maximised over time and phase) between two input waveforms.

    Parameters:
        wf1: First input waveform.
        wf2: Second input waveform.
        f_low: Lower bound of frequency integral.
        subsample_interpolation: Whether to use subsample interpolation.
        return_phase: Whether to return phase of maximum match.
        
    Returns:
        Amplitude (and optionally phase) of match.
    """

    # Resize the waveforms to the same length
    wf1, wf2 = resize_wfs([wf1, wf2])

    # Generate the aLIGO ZDHP PSD
    psd = gen_psd(wf1, f_low)

    # Perform match
    m = match(wf1.real(), wf2.real(), psd=psd, low_frequency_cutoff=f_low+3, subsample_interpolation=subsample_interpolation, return_phase=return_phase)

    # Additionally returns phase required to match waveforms up if requested
    if return_phase:
        return m[0], m[2]
    else:
        return m[0]

def overlap_cplx_wfs(wf1, wf2, f_low, normalized=True):
    """
    Calculates complex overlap (overlap maximised over phase) between two input waveforms.

    Parameters:
        wf1: First input waveform.
        wf2: Second input waveform.
        f_low: Starting frequency of waveforms.
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
    m = overlap_cplx(wf1.real(), wf2.real(), psd=psd, low_frequency_cutoff=f_low+3, normalized=normalized)

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

def get_comp_shifts(f_low, e, M, q, n, sample_rate, approximant):
    '''
    Calculates shifted frequency and eccentricity required to create each component
    waveform (beyond first).

    Parameters:
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        n: Number of waveform components.
        sample_rate: Sample rate of waveform.
        approximant: Approximant to use.

    Returns:
        Shifted frequency and eccentricity for all components beyond first.
    '''

    # Finds shifted frequency and eccentricity without correction
    max_s_f = shifted_f(f_low, e, M, q)
    s_f_vals = np.linspace(f_low, max_s_f, n, endpoint=False)[1:]
    s_e_vals = shifted_e(s_f_vals, f_low, e)

    return s_f_vals, s_e_vals

def gen_component_wfs(f_low, e, M, q, n, sample_rate, approximant, normalisation, phase):
    '''
    Creates n component waveforms used to make h_1,...,h_n, all equally spaced in
    true anomaly.
    
    Parameters:
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        n: Number of waveform components.
        sample_rate: Sample rate of waveform.
        approximant: Approximant to use.
        normalisation: Whether to normalise s_1,...,s_n components to ensure (sj|sj) is constant.
        phase: Initial phase of s_1,...,s_n components.
        
    Returns:
        Component waveforms.
    '''

    # Generates first (unshifted) component waveform and shifts required for others
    h = gen_wf(f_low, e, M, q, sample_rate, approximant, phase=phase)
    s_f_vals, s_e_vals = get_comp_shifts(f_low, e, M, q, n, sample_rate, approximant)

    # Tapers first waveform
    h = taper_wf(h)
    
    # Calculates normalisation factor using sigma function
    if normalisation:
        # Generate the aLIGO ZDHP PSD
        h.resize(ceiltwo(len(h))) 
        psd = gen_psd(h, f_low)
        sigma_0 = sigma(h.real(), psd=psd, low_frequency_cutoff=f_low+3)

    comp_wfs = [h]
    
    # Generate all component waveforms
    for i in range(n-1):

        # Create waveform
        h = gen_wf(s_f_vals[i], s_e_vals[i], M, q, sample_rate, approximant, phase=phase)

        # Trim waveform to same size as first (shortest), and corrects phase
        h = trim_wf(h, comp_wfs[0])
        overlap = overlap_cplx_wfs(h, comp_wfs[0], f_low)
        phase_angle = np.angle(overlap)/2
        h = gen_wf(s_f_vals[i], s_e_vals[i], M, q, sample_rate, approximant, phase=phase+phase_angle)
        h = trim_wf(h, comp_wfs[0])

        # Tapers
        h = taper_wf(h)
        
        # Normalises waveform if requested
        if normalisation:
            sigma_h = sigma(h.real(), psd=psd, low_frequency_cutoff=f_low+3)
            h *= sigma_0/sigma_h

        comp_wfs.append(h)

    return comp_wfs

def get_dominance_order(n):
    '''
    Creates indexing array to order h1, ..., hn waveforms from their natural roots of unity order 
    to their order of dominance.
    
    Parameters:
        n: Number of waveform components.
        
    Returns:
        Indexing array.
    '''

    # Always start with j=0
    j_order = [0]

    # Add increasing pairs of j and n-j
    for i in range(1, int((n+1)/2)):
        j_order.append(i)
        j_order.append(n-i)

    # Add n/2 if n is even
    if n%2 == 0:
        j_order.append(int(n/2))

    return j_order

def GS_proj(u, v, f_low, psd):
    '''
    Performs projection used in Grant-Schmidt orthogonalisation, defined as 
    u*(v|u)/(u|u).
    
    Parameters:
        u: Waveform u defined above.
        v: Waveform v defined above.
        f_low: Starting frequency.
        psd: Psd to use to weight complex overlap.
        
    Returns:
        Grant-Schmidt orthogonalised h1,...,hn.
    '''

    numerator = overlap_cplx(v.real(), u.real(), psd=psd, low_frequency_cutoff=f_low+3, normalized=False)
    denominator = overlap_cplx(u.real(), u.real(), psd=psd, low_frequency_cutoff=f_low+3, normalized=False)

    return u*numerator/denominator

def GS_orthogonalise(f_low, wfs):
    '''
    Performs Grant-Schmidt orthogonalisation on waveforms h1,...,hn to ensure 
    (hj|hm) = 0 for j!=m.
    
    Parameters:
        f_low: Starting frequency.
        wfs: Waveforms h1,...,hn.
        
    Returns:
        Grant-Schmidt orthogonalised h1,...,hn.
    '''

    # Generates psd for use in orthogonalisation
    psd = gen_psd(wfs[0], f_low)

    # Orthogonalises each waveform in turn
    for i in range(1,len(wfs)):
        for j in range(i):
            wfs[i] = wfs[i] - GS_proj(wfs[j], wfs[i], f_low, psd)

    return wfs

def get_h_TD(f_low, coeffs, comp_wfs, GS_normalisation):
    """
    Combines waveform components in time domain to form h1, ..., hn and h as follows:

    Parameters:
        f_low: Starting frequency.
        coeffs: List containing coefficients of h_1, ..., h_n.
        comp_wfs: Waveform components s_1, ..., s_n.
        GS_normalisation: Whether to perform Grant-Schmidt orthogonalisaton to ensure (hj|hm) = 0 for j!=m.
        
    Returns:
        All waveform components and combinations: h, h1, ..., h_n, s_1, ..., s_n
    """

    # Find first primitive root of unity
    prim_root = np.e**(2j*np.pi/len(coeffs))
    
    # Build h1, ..., hn
    hs = []
    for i in range(len(coeffs)):
        hs.append((1/len(coeffs))*comp_wfs[0])
        for j in range(len(coeffs)-1):
            hs[-1] += (1/len(coeffs))*comp_wfs[j+1]*prim_root**(i*(j+1))

    # Re-order by dominance rather than natural roots of unity order
    j_order = get_dominance_order(len(coeffs))
    hs = [hs[i] for i in j_order]

    # Perform Grant-Schmidt orthogonalisation if requested
    if GS_normalisation:
        hs = GS_orthogonalise(f_low, hs)

    # Calculates overall waveform using complex coefficients A, B, C, ...
    h = coeffs[0]*hs[0]
    for i in range(len(coeffs)-1):
        h += coeffs[i+1]*hs[i+1]
    
    # Returns overall waveform and components for testing purposes
    return h, *hs, *comp_wfs

def get_h(coeffs, f_low, e, M, q, sample_rate, approximant='TEOBResumS', subsample_interpolation=True, GS_normalisation=True, comp_normalisation=False, comp_phase=0):
    """
    Generates a overall h waveform, h_1,...h_n, and s_1,...,s_n.

    Parameters:
        coeffs: List containing coefficients of h_1,...,h_n.
        f_low: Starting frequency.
        e: Eccentricity.
        M: Total mass.
        q: Mass ratio.
        sample_rate: Sample rate of waveform.
        approximant: Approximant to use.
        subsample_interpolation: Whether to use subsample interpolation.
        GS_normalisation: Whether to perform Grant-Schmidt orthogonalisaton to ensure (hj|hm) = 0 for j!=m.
        comp_normalisation: Whether to normalise s_1,...,s_n components to ensure (sj|sj) is constant.
        comp_phase: Initial phase of s_1,...,s_n components.
        
    Returns:
        All waveform components and combinations: h, h1, ..., h_n, s_1, ..., s_n
    """

    # Other approximants are deprecated
    assert approximant == 'TEOBResumS'

    # Gets (normalised) components which make up overall waveform
    component_wfs = gen_component_wfs(f_low, e, M, q, len(coeffs), sample_rate, approximant, comp_normalisation, comp_phase)

    # Calculate overall waveform and components in time domain
    wfs = get_h_TD(f_low, coeffs, component_wfs, GS_normalisation)
   
    return wfs