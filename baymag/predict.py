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


def predict_mgca(seatemp, cleaning, spp, latlon, depth, seasonal_seatemp=False,
                 ph=None, omega=None, drawsfun=get_draws):
    """Predict Mg/Ca from sea temperature

    Parameters
    ----------
    seatemp : ndarray
        n-length array of sea temperature observations (°C) from a single
        location.
    cleaning : ndarray
        Binary n-length array indicating the cleaning method used for the
        inferred Mg/Ca series. ``1`` for reductive, ``0`` for BCP (Barker).
    spp : str
        Foraminifera species of the inferred Mg/Ca series.
    latlon : tuple of floats
        Latitude and longitude of site. Latitude must be between -90 and 90.
        Longitude between -180 and 180.
    depth : float
        Water depth (m). Increasing values indicate increasing depth below sea
        level.
    seasonal_seatemp : bool, optional
        Indicates whether sea-surface temperature is annual or seasonal
        estimate. If ``True``, ``spp`` must be specified.
    ph : float or None, optional
        Optional sea water pH. Estimated from sea surface if ``None``.
    omega : float or None, optional
        Optional sea water omega. Estimated from sea water at sample depth if
        ``None``.
    drawsfun : function-like, optional
        For debugging and testing. Object to be called to get MCMC model
        parameter draws. Don't mess with this.

    Returns
    -------
    out : MgCaPrediction
    """
    seatemp = np.array(seatemp)
    cleaning = np.array(cleaning)

    assert depth >= 0, 'sample `depth` should be positive'

    if omega is None:
        _, _, omega = carbion(latlon, depth=depth)

    if ph is None:
        ph, _, _ = carbion(latlon, depth=0)

    # Standardize pH and omega for model.
    ph -= 8
    omega = 1 / omega

    alpha, beta_temp, beta_ph, beta_omega, beta_clean, sigma = drawsfun(spp, seasonal_seatemp)

    mgca = np.empty((len(seatemp), len(sigma)))
    mgca[:] = np.nan

    for i in range(len(sigma)):
        alpha_now = alpha[i]
        beta_temp_now = beta_temp[i]
        beta_omega_now = beta_omega[i]
        beta_clean_now = beta_clean[i]
        beta_ph_now = beta_ph[i]
        sigma_now = sigma[i]
        clean_term = (1 - beta_clean_now * cleaning)
        mu = ((alpha_now + np.exp(beta_temp_now * seatemp + ph * beta_ph_now)
               + beta_omega_now * omega) * clean_term)
        mgca[:, i] = np.random.normal(mu, sigma_now)

    out = MgCaPrediction(ensemble=mgca, spp=spp)

    return out
