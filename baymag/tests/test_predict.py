import numpy as np

import baymag.predict


def test_percentile():
    prediction_test = baymag.predict.Prediction(ensemble=np.array([range(10), range(10)]),
                                                spp='ruber_w')
    victim = prediction_test.percentile()
    goal = np.array([[0, 0], [4, 4], [9, 9]]).T
    np.testing.assert_equal(victim, goal)


def test_predict_mgca():
    np.random.seed(123)
    victim = baymag.predict.predict_mgca(seatemp=np.array([10, 20, 30]),
                                         cleaning=np.array([1] * 3),
                                         spp='ruber',
                                         salinity=35.0,
                                         ph=8.1,
                                         omega=0.85,
                                         )
    goal_median = np.array([1.143593, 2.158549, 4.016868])
    np.testing.assert_allclose(np.median(victim.ensemble, axis=1),
                               goal_median, rtol=0, atol=1e-4)


def test_sw_correction():
    p = baymag.predict.MgCaPrediction(ensemble=np.ones((2, 3)),
                                      spp='abc')

    def debugdraws():
        return np.array([0.0, 1.0]), np.array([[1, 1, 1], [2, 2, 2]])

    test2 = baymag.predict.sw_correction(p, 0.5, debugdraws)
    goal = np.ones(p.ensemble.shape) * 1.5
    np.testing.assert_allclose(test2.ensemble, goal, atol=1e-16, rtol=0)
