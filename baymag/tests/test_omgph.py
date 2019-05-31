import pytest
import numpy as np

import baymag.omgph



def test_get_ph():
    """General get_ph test case"""
    goal = 8.113039017
    latlon = (29.81, -43.22)
    victim = baymag.omgph.fetch_ph(latlon=latlon)
    np.testing.assert_allclose(victim, goal, atol=1e-3, rtol=0)


def test_get_omega():
    """General get_omega test case"""
    goal = 1.0978
    latlon = (17.3, -48.4)
    depth = 3975
    victim = baymag.omgph.fetch_omega(latlon=latlon, depth=depth)
    np.testing.assert_allclose(victim, goal, atol=1e-3, rtol=0)


def test_get_omega_caribbean():
    """Test ofr site in Caribbean"""
    goal = 1.5313
    latlon = (20.42, -80.14)
    depth = 2330
    victim = baymag.omgph.fetch_omega(latlon=latlon, depth=depth)
    np.testing.assert_allclose(victim, goal, atol=1e-3, rtol=0)


def test_get_omega_gom():
    """Test from site in Gulf of Mexico"""
    goal = 1.8104
    latlon = (23.2, -90.0)
    depth = 599
    victim = baymag.omgph.fetch_omega(latlon=latlon, depth=depth)
    np.testing.assert_allclose(victim, goal, atol=1e-3, rtol=0)
