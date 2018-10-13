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
