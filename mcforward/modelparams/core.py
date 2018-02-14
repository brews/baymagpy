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


def get_draws(spp):
    raise NotImplementedError
