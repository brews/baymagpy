import pytest
import numpy as np

import baymag.omega


def test_get_omega():
    """General get_omega test case"""
    goal = 1.112018219432555
    latlon = (17.3, -48.4)
    depth = 3975
    victim = baymag.omega.get_omega(latlon=latlon, depth=depth)
    np.testing.assert_allclose(victim, goal, atol=1e-5)


def test_get_omega_scs():
    """Test for site in South China Sea"""
    goal = 1.209596401825736
    latlon = (15.463, 114.398)
    depth = 1446
    victim = baymag.omega.get_omega(latlon=latlon, depth=depth)
    np.testing.assert_allclose(victim, goal, atol=1e-4)


def test_get_omega_caribbean():
    """Test ofr site in Caribbean"""
    goal = 1.620837849676754
    latlon = (20.42, -80.14)
    depth = 2330
    victim = baymag.omega.get_omega(latlon=latlon, depth=depth)
    np.testing.assert_allclose(victim, goal, atol=1e-4)


def test_get_omega_mediterranean():
    """Test ofr site in Mediterranean"""
    goal = 3.247495785049527
    latlon = (34.53, 17.98)
    depth = 3402
    victim = baymag.omega.get_omega(latlon=latlon, depth=depth)
    np.testing.assert_allclose(victim, goal, atol=1e-4)


def test_get_omega_gom():
    """Test from site in Gulf of Mexico"""
    goal = 1.810377376823078
    latlon = (23.2, -90.0)
    depth = 599
    victim = baymag.omega.get_omega(latlon=latlon, depth=depth)
    np.testing.assert_allclose(victim, goal, atol=1e-4)


def test_get_omega_arctic():
    """Test from site in Arctic"""
    goal = 1.205628075781792
    latlon = (67.9, -4.0)
    depth = 3676
    victim = baymag.omega.get_omega(latlon=latlon, depth=depth)
    np.testing.assert_allclose(victim, goal, atol=1e-4)
