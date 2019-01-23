import attr
import numpy as np

from baymag.omega import omgph
from baymag.modelparams import get_draws
from baymag.modelparams import get_sw_draws


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


def predict_mgca(seatemp, cleaning, spp=None, seasonal_seatemp=False,
                 omega=None, latlon=None, depth=None, sw_age=None,
                 drawsfun=get_draws):
    """Predict Mg/Ca from sea temperature

    Parameters
    ----------
    seatemp : ndarray
        n-length array of sea temperature observations (°C) from a single
        location.
    cleaning : ndarray
        Binary n-length array indicating the cleaning method used for the
        inferred Mg/Ca series. ``1`` for reductive, ``0`` for BCP (Barker).
    spp : str or None, optional
        Foraminifera species of the inferred Mg/Ca series, using hierarchical
        calibration model parameters. Default is None, which uses pooled
        calibration model parameters.
    seasonal_seatemp : bool, optional
        Indicates whether sea-surface temperature is annual or seasonal
        estimate. If True, ``spp`` must be specified. Default is False.
    omega : float or None, optional
        Optional sea water omega. Estimated from modern seawater at sample
        ``depth`` and ``latlon`` if None. Default is None.
    latlon : tuple of floats or None, optional
        Latitude and longitude of site. Used to find sample omega. Latitude
        must be between -90 and 90. Longitude between -180 and 180. Must be
        given if ``omega`` is None. Default is None.
    depth : float or None, optional
        Sample waterdepth (m). Increasing values indicate increasing depth below sea
        level. Used to find sample omega. Required arg if ``omega`` not given.
        Default is None.
    sw_age : ndarray or None, optional
        Optional n-length sequence indicating the age of values in ``seatemp``
        to apply Mg/Ca correction for Deep Time seawater. Units must be Ma.
        Default argument ``None`` does not apply Mg/Ca seawater correction.
    drawsfun : function-like, optional
        For debugging and testing. Object to be called to get MCMC model
        parameter draws. Don't mess with this.

    Returns
    -------
    out : MgCaPrediction
    """
    seatemp = np.atleast_1d(seatemp)
    cleaning = np.atleast_1d(cleaning)

    if omega is None:
        assert (depth is not None) and (latlon is not None), '`depth` and `latlon` need to be given when `omega` is None'
        assert depth >= 0, 'sample `depth` should be positive'
        omega = omgph(latlon, depth=depth)

    # Invert omega for model.
    omega = 1 / omega
    omega = np.atleast_1d(omega)

    alpha, beta_temp, beta_omega, beta_clean, sigma = drawsfun(spp, seasonal_seatemp)

    clean_term = (1 - beta_clean * cleaning[:, np.newaxis])
    mu = ((alpha + np.exp(beta_temp * seatemp[:, np.newaxis]) + beta_omega
           * omega[:, np.newaxis]) * clean_term)
    mgca = np.random.normal(mu, sigma)

    out = MgCaPrediction(ensemble=mgca, spp=spp)

    if sw_age is not None:
        out = sw_correction(out, age=sw_age)

    return out


def sw_correction(mgcaprediction, age, drawsfun=None):
    """Apply Deep Time seawater correction to Mg/Ca prediction.

    Parameters
    ----------
    mgcaprediction : baymag.predict.MgCaPrediction
    age : sequence-like
        Age of predictions in ``prediction``. Must be in units Ma. n-length
        sequence where n == prediction.ensemble.shape[0].
    drawsfun : None or function-like, optional
        Optional function-like returning 2d array of MCMC parameter draws to
        use for seawater correction. Used for testing and debugging only.
        Default ``None`` uses ``baymag.modelparams.get_sw_draws()``.

    Returns
    -------
    out : baymag.MgCaPrediction
        Copy of mgcaprediction with correction to ensemble.
    """
    if drawsfun is None:
        beta_draws = get_sw_draws()
    else:
        beta_draws = drawsfun()

    age = np.asanyarray(age)
    mgsw = 1 / (beta_draws[0] * age[:, np.newaxis] + beta_draws[1])

    # ratio to modern value
    mgsw /= mgsw[0]

    out = MgCaPrediction(ensemble=np.array(mgcaprediction.ensemble * mgsw),
                         spp=str(mgcaprediction.spp))
    return out
