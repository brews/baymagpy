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
    goal_median = np.array([0.752568015634634,
                            2.006171193280855,
                            4.384751781887797])
    np.testing.assert_allclose(np.median(victim.ensemble, axis=1), goal_median,
                               atol=1e-3)
