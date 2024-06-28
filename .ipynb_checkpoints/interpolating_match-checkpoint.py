import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator
from scipy.optimize import minimize

def find_min_max(data):
    """
    Finds minimum and maximum match of h0, h1, h1/h0 across varying mean anomaly.

    Parameters:
        data: Dictionary containing matches.

    Returns:
        data: Dictionary containing matches with min/max matches added.
    """

    # Loop over each chirp mass grid
    for chirp in data.keys():

        # Loop over h0, h1, h1/h0
        for key in list(data[chirp].keys()):
            
            # Find min/max vals and add to grid
            if key[0] == 'h' or key=='quad':
                data[chirp][f'{key}_max'] = np.max(np.array(data[chirp][key]), axis=1)
                data[chirp][f'{key}_min'] = np.min(np.array(data[chirp][key]), axis=1)

    return data

def create_2D_interps(data, param_vals=np.linspace(0, 0.2, 101)):
    """
    Create interpolation objects which give the min and max match value at 
    arbitrary chirp mass and point in parameter space on line of degeneracy.

    Parameters:
        data: Dictionary containing matches.
        param_vals: Array of eccentricity values used to create data.

    Returns:
        interp_objs: Created interpolation objects.
    """

    interp_objs = []

    # Get param vals from the data if available
    first_chirp = list(data.keys())[0]
    if 'e_vals' in data[first_chirp].keys():
        param_vals = data[first_chirp]['e_vals']
    
    # Loop over h0, h1, h1/h0:
    for key in ['h0', 'h1', 'h1_h0']:

        max_vals = []
        min_vals = []
        
        # Loop over each chirp mass grid to get all max/min match values
        for chirp in data.keys():
            max_vals.append(data[chirp][f'{key}_max'])
            min_vals.append(data[chirp][f'{key}_min'])
        max_vals = np.array(max_vals).flatten()
        min_vals = np.array(min_vals).flatten()

        # Get flat chirp mass, param values arrays
        ecc_vals, chirp_vals = np.meshgrid(param_vals, list(data.keys()))
        ecc_vals = ecc_vals.flatten()
        chirp_vals = chirp_vals.flatten()
        
        # Create max/min interpolation objects
        max_interp = LinearNDInterpolator(list(zip(chirp_vals, ecc_vals)), max_vals)
        min_interp = LinearNDInterpolator(list(zip(chirp_vals, ecc_vals)), min_vals)
        interp_objs.append([max_interp, min_interp])

    return interp_objs

def find_ecc_range(match, chirp, interps, slope='increasing', max_ecc=0.4):
    """
    Find range of eccentricities corresponding to match value.

    Parameters:
        match: Match value.
        chirp: Eccentric chirp mass of degeneracy line.
        interps: Interpolation objects to use.
        slope: Slope direction of match against ecc along degeneracy line.
        max_ecc: Maximum value of eccentricity used to create interpolation objects.

    Returns:
        min_ecc: Minimum eccentricity.
        max_ecc: Maximum eccentricity.
    """

    # Check whether in range of each interp, deal with edge cases
    ecc_range = np.arange(0, max_ecc+0.001, 0.001)
    line_eccs = [0,0]
    slope_dict = {'increasing': 0, 'decreasing': 1, None: 2}
    slope_ind = slope_dict[slope]
    for i in range(2):
        interp_arr = interps[i](chirp, ecc_range)
        edge_cases = [[[ecc_range[np.argmax(interp_arr)],0,None],[ecc_range[np.argmax(interp_arr)],1,None],[None, None, None]],
                      [[1,ecc_range[np.argmin(interp_arr)],None],[0,ecc_range[np.argmin(interp_arr)],None],[None, None, None]]]
        if match > np.max(interp_arr):
            range_ind = 0
        elif match < np.min(interp_arr):
            range_ind = 1
        else:
            range_ind = 2
        line_eccs[i] = edge_cases[i][slope_ind][range_ind]
        
    # Find eccentricities corresponding to max and min lines
    for i in range(2):
        if line_eccs[i] is None:
            max_result = minimize(lambda x: abs(interps[i](chirp, x) - match), max_ecc/2, bounds=[(0, max_ecc)], method='Powell')
            line_eccs[i] = max_result['x'][0]
    
    # Identify low and high of eccentricity range
    max_ecc = np.max(line_eccs)
    min_ecc = np.min(line_eccs)

    return min_ecc, max_ecc

def find_ecc_range_samples(matches, chirp, interps, max_ecc=0.4):
    """
    Find range of eccentricities corresponding to match values of samples. Assumes
    slope is increasing.

    Parameters:
        matches: Match values.
        chirp: Eccentric chirp mass of degeneracy line.
        interps: Interpolation objects to use.
        max_ecc: Maximum value of eccentricity used to create interpolation objects.

    Returns:
        min_ecc: Minimum eccentricity.
        max_ecc: Maximum eccentricity.
    """

    # Ensure matches is numpy array
    matches = np.array(matches)

    # Create reverse interpolation object to get the eccentricity from match value
    ecc_range = np.arange(0, max_ecc+0.001, 0.001)
    max_interp_arr = interps[0](chirp, ecc_range)
    min_interp_arr = interps[1](chirp, ecc_range)
    max_interp = interp1d(max_interp_arr, ecc_range)
    min_interp = interp1d(min_interp_arr, ecc_range)

    # Check whether in range of each interp, deal with edge cases
    ecc_arr = np.array([np.full_like(matches, 5)]*2)
    ecc_arr[0][matches<np.min(max_interp_arr)] = 0
    ecc_arr[0][matches>np.max(max_interp_arr)] = ecc_range[np.argmax(max_interp_arr)]
    ecc_arr[1][matches<np.min(min_interp_arr)] = ecc_range[np.argmin(min_interp_arr)]
    ecc_arr[1][matches>np.max(min_interp_arr)] = 1
    
    # Find eccentricities corresponding to max and min lines
    ecc_arr[0][ecc_arr[0]==5] = max_interp(matches[ecc_arr[0]==5])
    ecc_arr[1][ecc_arr[1]==5] = min_interp(matches[ecc_arr[1]==5])

    return ecc_arr