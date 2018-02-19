import pytest
import xarray as xr
import numpy as np

from mcforward.omega.utils import get_nearest, chord_distance, DistanceThresholdError


@pytest.fixture
def latlondepth_da():
    da = xr.DataArray(np.arange(8).reshape((2, 2, 2)),
                      coords=[('depth', [0, 1]), ('lat', [17.5, -69.5]),
                              ('lon', [-48.5, -179.5])],
                      dims=['depth', 'lat', 'lon'])
    return da


@pytest.mark.parametrize("test_input,expected", [
    ([(17.3, -48.4), (17.5, -48.5)], 24.668176282908473),
    ([(17.3, -48.4), [(17.5, -48.5), (-69.5, -179.5)]],
     np.array([[24.668176282908473], [1.104116337324761e+04]])),
])
def test_chord_distance(test_input, expected):
    """Test chord_distance against inputs of multiple size"""
    victim = chord_distance(*test_input)
    np.testing.assert_allclose(victim, expected, atol=1e-8)


def test_get_nearest_latlon(latlondepth_da):
    """Simple test of get_nearest for latlon"""
    victim = get_nearest(latlon=(17.3, -48.4), dain=latlondepth_da)

    np.testing.assert_equal(victim.values, np.array([0, 4]))
    np.testing.assert_equal(victim.depth.values, np.array([0, 1]))
    assert victim.yx.to_dict()['data'] == (17.5, -48.5)


def test_get_nearest_depth(latlondepth_da):
    """Simple test of get_nearest for latlon with depth"""
    victim = get_nearest(latlon=(17.3, -48.4), dain=latlondepth_da, depth=0.7)

    np.testing.assert_equal(victim.values, np.array([4]))
    np.testing.assert_equal(victim.depth.values, np.array([1]))
    assert victim.yx.to_dict()['data'] == (17.5, -48.5)


def test_get_nearest_badthreshold(latlondepth_da):
    """Ensure get_nearest throws error if over threshold"""
    with pytest.raises(DistanceThresholdError):
        get_nearest(latlon=(17.3, -48.4), dain=latlondepth_da, depth=0.7,
                    distance_threshold=1)
