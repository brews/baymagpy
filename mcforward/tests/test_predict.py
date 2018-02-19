import pytest
import numpy as np

import mcforward.predict


def test_percentile():
    prediction_test = mcforward.predict.Prediction(ensemble=np.array([range(10), range(10)]),
                                                   spp='ruber')
    victim = prediction_test.percentile()
    goal = np.array([[0, 0], [4, 4], [9, 9]]).T
    np.testing.assert_equal(victim, goal)


def test_predict_mgca():
    np.random.seed(123)
    victim = mcforward.predict.predict_mgca(seatemp=np.array([10, 20, 30]),
                                            cleaning=np.array([1] * 3),
                                            spp='ruber',
                                            latlon=(17.3, -48.4),
                                            depth=3975)
    goal_median = np.array([0.807673971591095,
                            2.093295984518164,
                            4.529007834393901])
    np.testing.assert_allclose(np.round(np.median(victim.ensemble, axis=1), 1),
                               np.round(goal_median, 1),
                               atol=1e-1)
