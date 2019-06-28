"""Code to grab dumped MCMC parameter posterior trace draws.
"""


__all__ = ['get_draws', 'get_sw_draws']


from os import path
import numpy as np
from baymag.utils import get_matlab_resource


MGSW_POST = get_matlab_resource(path.join('modelparams', 'mgsw_gaussian.mat'),
                                squeeze_me=True)
POOLED_PARAMS = get_matlab_resource(path.join('modelparams', 'pooled_model_params.mat'),
                                squeeze_me=True)
POOLEDSEA_PARAMS = get_matlab_resource(path.join('modelparams', 'pooled_sea_model_params.mat'),
                                squeeze_me=True)
SPP_PARAMS = get_matlab_resource(path.join('modelparams', 'species_model_params.mat'),
                                squeeze_me=True)


def get_sw_draws():
    """Get copy of arrays for Deep Time Mg/Ca seawater correction.

    Returns
    -------
    xt : ndarray
    mgsmooth : ndarray
    """
    mgsmooth = np.array(MGSW_POST['mg_smooth']).copy()
    xt = np.array(MGSW_POST['xt']).copy()
    return xt, mgsmooth


def get_draws(species):
    """Get MCMC parameter draws

    Parameters
    ----------
    species : str
        One of 'all', 'all_sea', 'bulloides', 'ruber', 'pachy', 'sacculifer'.

    Returns
    -------
    alpha : ndarray
    beta_temp : ndarray
    beta_salinity : ndarray
    beta_omega : ndarray
    beta_ph : ndarray
    beta_clean : ndarray
    sigma : ndarray
    """
    species = str(species)
    species_old = species

    draws_map = {'all': (POOLED_PARAMS, None),
                 'all_sea': (POOLEDSEA_PARAMS, None),
                 'ruber': (SPP_PARAMS, 0),
                 'bulloides': (SPP_PARAMS, 1),
                 'sacculifer': (SPP_PARAMS, 2),
                 'pachy': (SPP_PARAMS, 3),
                 }

    foram_map = {'G. bulloides': 'bulloides',
                 'N. pachyderma sinistral': 'pachy',
                 'N. incompta': 'pachy',
                 'G. ruber pink': 'ruber',
                 'G. ruber white': 'ruber',
                 'G. ruber': 'ruber',
                 'G. sacculifer': 'sacculifer',
                 }
    # Translate old species to new species names, for legacy support.
    species = foram_map.get(species, str(species))

    assert species in draws_map.keys(), f'"{species_old}" must be one of draws_map {list(draws_map.keys())}'

    draws, idx = draws_map[species]

    beta_temp = draws['betaT'].copy()
    beta_salinity = draws['betaS'].copy()
    beta_omega = draws['betaO'].copy()
    beta_clean = draws['betaC'].copy()
    beta_ph = draws['betaP'].copy()
    sigma = draws['sigma'][:, idx].copy()
    alpha = draws['alpha'][:, idx].copy()

    return alpha, beta_temp, beta_salinity, beta_omega, beta_ph, beta_clean, sigma
