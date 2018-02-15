import os.path
from copy import deepcopy
import attr
import numpy as np

from mcforward.utils import get_matlab_resource


@attr.s
class Draws:
    alpha = attr.ib()
    beta1 = attr.ib()
    beta2 = attr.ib()
    beta3 = attr.ib()
    sigma = attr.ib()

d = get_matlab_resource('/modelparams/ruber_stan_posterior.mat')
model_draws = {'ruber': Draws(alpha=d['alpha'].ravel(),
                              beta1=d['betaT'].ravel(),
                              beta2=d['betaO'].ravel(),
                              beta3=d['betaC'].ravel(),
                              sigma=d['sigma'].ravel())}

def get_draws(spp):
    """Get model parameter draws for spp"""
    assert spp in ['ruber']
    return deepcopy(model_draws['ruber'])
