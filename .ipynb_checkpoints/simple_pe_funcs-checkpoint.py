import numpy as np
from pycbc.filter.matchedfilter import matched_filter

def calculate_mode_snr(strain_data, ifo_psd, waveform_modes, t_start, t_end,
                       f_low, modes, dominant_mode='22'):
    """
    Calculate the SNR in each of the modes.  This is done by finding time of
    the peak SNR for the dominant mode, and then calculating the SNR of other
    modes at that time.

    Parameters
    ----------
    strain_data: pycbc.Time_Series
        the ifo data
    ifo_psd: pycbc.Frequency_Series
        PSD for ifo
    waveform_modes: dict
        dictionary of waveform modes (time/frequency series)
    t_start: float
        beginning of time window to look for SNR peak
    t_end: float
        end of time window to look for SNR peak
    f_low: float
        low frequency cutoff
    modes: list
        the modes to calculate SNR for
    dominant_mode: str
        mode that is used to define the peak time

    Returns
    -------
    z: dict
        dictionary of complex SNRs for each mode
    t: float
        the time of the max SNR
    """

    if dominant_mode not in waveform_modes.keys():
        print("Please give the waveform for the dominant mode")
        return

    s = matched_filter(waveform_modes[dominant_mode], strain_data, ifo_psd,
                       low_frequency_cutoff=f_low)
    snr = s.crop(t_start - s.start_time, s.end_time - t_end)

    # find the peak and use this for the other modes later
    i_max = snr.abs_arg_max()
    t_max = snr.sample_times[i_max]

    z = {}
    for mode in modes:
        s = matched_filter(waveform_modes[mode], strain_data, psd=ifo_psd,
                           low_frequency_cutoff=f_low,
                           high_frequency_cutoff=ifo_psd.sample_frequencies[-1],
                           sigmasq=None)
        snr_ts = s.crop(t_start - s.start_time, s.end_time - t_end)
        z[mode] = snr_ts[i_max]

    return z, t_max

def network_mode_snr(z, ifos, modes, dominant_mode='22'):
    """
    Calculate the Network SNR in each of the specified modes.  For the
    dominant mode, this is simply the root sum square of the snrs in each
    ifo.  For the other modes, we calculate both the rss SNR and the network
    SNR which requires the relative phase between ifos is consistent with
    the dominant.

    Parameters
    ----------
    z: dict
        dictionary of dictionaries of SNRs in each mode (in each ifo)
    ifos: list
        A list of ifos to use
    modes: list
        A list of modes to use
    dominant_mode: str
        the mode with most power (for orthogonalization)

    Returns
    -------
    rss_snr: dict
        the root sum squared SNR in each mode
    net_snr: dict
        the SNR in each mode that is consistent (in amplitude and
        phase) with the dominant mode SNR
    """

    z_array = {}

    rss_snr = {}

    for mode in modes:
        z_array[mode] = np.array([z[ifo][mode] for ifo in ifos])
        rss_snr[mode] = np.linalg.norm(z_array[mode])

    net_snr = {}

    for mode in modes:
        net_snr[mode] = np.abs(np.inner(z_array[dominant_mode],
                                        z_array[mode].conjugate())) / \
                        rss_snr[dominant_mode]

    return rss_snr, net_snr