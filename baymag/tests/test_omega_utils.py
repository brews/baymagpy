import pytest
import xarray as xr
import numpy as np

from baymag.omega.core import chord_distance


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
