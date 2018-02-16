import pytest
import numpy as np

from mcforward.omega.utils import get_nearest, chord_distance


@pytest.mark.parametrize("test_input,expected", [
    ([(17.3, -48.4), (17.5, -48.5)], 24.668176282908473),
    ([(17.3, -48.4), [(17.5, -48.5), (-69.5, -179.5)]],
     np.array([[24.668176282908473], [1.104116337324761e+04]])),
])
def test_chord_distance(test_input, expected):
    """Test chord_distance against inputs of multiple size"""
    victim = chord_distance(*test_input)
    np.testing.assert_allclose(victim, expected, atol=1e-8)


def test_get_nearest():
    raise NotImplementedError
    # mcforward.omega.utils.get_nearest()
