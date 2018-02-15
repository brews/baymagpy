import pytest
import numpy as np

import mcforward.omega


def test_get_omega():
    goal = 1.109015460859011
    victim = mcforward.omega.get_omega(latlon=(17.3, -48.4), depth=3975)
    np.testing.assert_allclose(victim, goal, atol=1e-5)
