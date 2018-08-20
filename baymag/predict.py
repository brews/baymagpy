import attr
import numpy as np

from baymag.omega import carbion
from baymag.modelparams import get_draws


@attr.s
class Prediction:
    """MCMC prediction

    Parameters
    ----------
    ensemble : ndarray
        Ensemble of predictions. A 2d array (nxm) for n predictands and m
        ensemble members.
    spp : str
        Foraminifera species used in prediction.
    """
    ensemble = attr.ib()
    spp = attr.ib()

    def percentile(self, q=None, interpolation='nearest'):
        """Compute the qth ranked percentile from ensemble members.

        Parameters
        ----------
        q : float, sequence of floats, or None, optional
            Percentiles (i.e. [0, 100]) to compute. Default is 5%, 50%, 95%.
        interpolation : str, optional
            Passed to numpy.percentile. Default is 'nearest'.

        Returns
        -------
        perc : ndarray
            A 2d (nxm) array of floats where n is the number of predictands in
            the ensemble and m is the number of percentiles (``len(q)``).
        """
        if q is None:
            q = [5, 50, 95]
        q = np.array(q, dtype=np.float64, copy=True)

        perc = np.percentile(self.ensemble, q=q, axis=1,
                             interpolation=interpolation)
        return perc.T


@attr.s
class MgCaPrediction(Prediction):
    pass


def predict_mgca(seatemp, cleaning, spp, latlon, depth):
    """Predict Mg/Ca from sea temperature

    Parameters
    ----------
    seatemp : ndarray
        n-length array of sea temperature observations (°C) from a single
        location.
    cleaning : ndarray
        n-length array indicating the cleaning method used for the inferred
        Mg/Ca series. ``1`` for BCP, ``2`` for reductive.
    spp : str
        Foraminifera species of the inferred Mg/Ca series.
    latlon : tuple of floats
        Latitude and longitude of site. Latitude must be between -90 and 90.
        Longitude between -180 and 180.
    depth : float
        Water depth (m).

    Returns
    -------
    out : MgCaPrediction
    """

    ph, delta_co3, omega = carbion(latlon, depth=depth)
    draws = get_draws(spp)

    mgca = np.empty((len(seatemp), len(draws.sigma)))
    mgca[:] = np.nan

    for i, sigma_now in enumerate(draws.sigma):
        alpha_now = draws.alpha[i]
        beta1_now = draws.beta1[i]
        beta2_now = draws.beta2[i]
        beta3_now = draws.beta3[i]
        clean_term = (1 - beta3_now * cleaning)
        mu = (alpha_now + np.exp(beta1_now * seatemp) + beta2_now * omega) * clean_term
        mgca[:, i] = np.random.normal(mu, sigma_now)

    out = MgCaPrediction(ensemble=mgca, spp=spp)

    return out
