import numpy as np
import baymag.predict


def test_percentile():
    prediction_test = baymag.predict.Prediction(ensemble=np.array([range(10), range(10)]),
                                                spp='ruber_w')
    victim = prediction_test.percentile()
    goal = np.array([[0, 0], [4, 4], [9, 9]]).T
    np.testing.assert_equal(victim, goal)


def test_predict_mgca():
    # TODO(brews): This is a very very rough integration test. Should do proper
    # test taking advantage of the `drawsfun` args.
    np.random.seed(123)
    victim = baymag.predict.predict_mgca(seatemp=np.array([10, 20, 30]),
                                         cleaning=np.array([1] * 3),
                                         spp='ruber_w',
                                         latlon=(17.3, -48.4),
                                         depth=3975)
    goal_median = np.array([1.7, 2.8, 4.5])
    np.testing.assert_allclose(np.round(np.median(victim.ensemble, axis=1), 1),
                               np.round(goal_median, 1),
                               atol=1e-1)


def test_sw_correction():
    test1 = baymag.predict.MgCaPrediction(ensemble=np.ones((2, 3)),
                                          spp='abc')
    test2 = baymag.predict.sw_correction(test1, [1, 2],
                                         drawsfun=lambda: np.array([[1, 2, 3],
                                                                    [0.5, 1, 2]]))
    goal = np.array([[1., 1., 1.], [0.6, 0.6, 0.625]])
    np.testing.assert_allclose(test2.ensemble, goal, atol=1e-10)


def test_predict_mgca_deeptime():
    """Integration test for sw_correction with predict_mgca"""
    # TODO(brews): This is a very very rough integration test. Should do proper
    # test taking advantage of the `drawsfun` args.
    np.random.seed(123)
    victim = baymag.predict.predict_mgca(seatemp=np.array([10, 20, 30]),
                                         cleaning=np.array([1] * 3),
                                         spp='ruber_w',
                                         latlon=(17.3, -48.4),
                                         depth=3975, sw_age=[1, 2, 3])
    goal_median = np.array([1.8, 2.6, 4.1])
    np.testing.assert_allclose(np.round(np.median(victim.ensemble, axis=1), 1),
                               np.round(goal_median, 1),
                               atol=1e-1)
