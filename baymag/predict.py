"""Code to make ``baymag`` predictions.
"""


__all__ = ['predict_mgca', 'sw_correction']


import attr
import numpy as np

from baymag.modelparams import get_draws
from baymag.modelparams import get_sw_draws
from baymag.modelparams import get_mgsw_smooth

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


def predict_mgca(seatemp, cleaning, salinity, ph, omega, spp, drawsfun=get_draws):
    """Predict Mg/Ca from sea temperature

    Parameters
    ----------
    seatemp : ndarray
        n-length array of sea temperature observations (°C) from a single
        location.
    cleaning : ndarray
        Binary n-length array indicating the cleaning method used for the
        inferred Mg/Ca series. ``1`` for reductive, ``0`` for BCP (Barker).
    salinity : scalar or ndarray
        Sea water salinity (PSU).
    ph : scalar or ndarray
        Sea water pH.
    omega : scalar or ndarray
        Sea water calcite saturation state.
    spp : str
        Calibration model parameter options. Must be one of:
        'all' : Pooled calibration using annual SSTs.
        'all_sea' : Pooled calibration using seasonal SSTs.
        'ruber' : Hierarchical calibration with G. ruber (white or pink).
        'bulloides' : Hierarchical calibration with G. bulloides.
        'sacculifer' : Hierarchical calibration with G. sacculifer.
        'pachy' : Hierarchical calibration with N. pachyderma or N. incompta.
    drawsfun : function-like, optional
        For debugging and testing. Object to be called to get MCMC model
        parameter draws. Don't mess with this.

    Returns
    -------
    out : MgCaPrediction

    See Also
    --------
    fetch_omega : Calculate modern insitu calcite saturation state (omega)
    fetch_ph : Fetch modern seawater surface insitu pH
    sw_correction : Apply Deep-Time seawater correction to Mg/Ca predictions
    """
    seatemp = np.atleast_1d(seatemp)
    cleaning = np.atleast_1d(cleaning)
    salinity = np.atleast_1d(salinity)
    ph = np.atleast_1d(ph)
    
    nlen = np.size(seatemp)
    # Invert omega for model.
    omega = omega ** -2
    omega = np.atleast_1d(omega)

    alpha, beta_temp, beta_salinity, beta_omega, beta_ph, beta_clean, sigma = drawsfun(spp)

    clean_term = (1 - beta_clean * cleaning[:, np.newaxis])
    
    if spp in ['all', 'all_sea']:
        alphaadj = np.tile(alpha,nlen)
        
        mu = (np.transpose(alphaadj) + beta_temp * seatemp[:, np.newaxis] + beta_omega * omega[:, np.newaxis]
              + beta_salinity * salinity[:, np.newaxis] + clean_term)
        
        mgca = np.exp(np.random.normal(mu, np.transpose(np.tile(sigma,nlen))))
        
    else:
        mu = (alpha + beta_temp * seatemp[:, np.newaxis] + beta_omega * omega[:, np.newaxis]
              + beta_salinity * salinity[:, np.newaxis] + clean_term)

        #if spp != 'pachy': # RT: this is not right ... see next 2 lines
        # if other than pachy or sacculifer, take sensitivity to pH into account
        if spp not in ['pachy', 'sacculifer']:
            mu += beta_ph * ph[:, np.newaxis]
            
        mgca = np.exp(np.random.normal(mu, sigma))
        
    out = MgCaPrediction(ensemble=mgca, spp=spp)

    return out


def sw_correction(mgcaprediction, age, drawsfun=None):
    """Apply Deep-Time seawater correction to Mg/Ca prediction.

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

    mgsw_smooth = get_mgsw_smooth()

    t = int(age * 2)
    
    mgsw = np.divide(mgsw_smooth[t,:] , mgsw_smooth[0,:])
    
    out = MgCaPrediction(ensemble=np.array(mgcaprediction.ensemble * mgsw),
                         spp=str(mgcaprediction.spp))
    return out
