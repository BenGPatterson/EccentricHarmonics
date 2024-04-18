import EOBRun_module
import numpy as np
import scipy.constants as const
import astropy.constants as aconst
from pycbc.waveform import get_td_waveform
from pycbc.types import timeseries
from scipy.optimize import minimize

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
    s_f = np.array(s_f).flatten()
    e = np.array(e).flatten()

    # Compute shifted eccentricity
    constant = shifted_e_const(f, e)
    init_guess = shifted_e_approx(s_f, f, e)
    bounds = [(0, 0.999)]
    best_fit = minimize(lambda x: np.sum(abs(shifted_e_const(s_f, x)-constant)), init_guess, bounds=bounds)
    s_e = np.array(best_fit['x'])

    return s_e