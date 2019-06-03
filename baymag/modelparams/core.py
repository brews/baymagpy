"""Code to grab dumped MCMC parameter posterior trace draws.
"""


__all__ = ['get_draws', 'get_sw_draws']


from os import path
import numpy as np

from baymag.utils import get_csv_resource, get_matlab_resource


POOLEDANNTRACE_PATH = path.join('modelparams', 'tracedumps', 'pooledann.csv')
POOLEDSEASTRACE_PATH = path.join('modelparams', 'tracedumps', 'pooledsea.csv')
HIERANNTRACE_PATH = path.join('modelparams', 'tracedumps', 'hierann.csv')
HIERSEASTRACE_PATH = path.join('modelparams', 'tracedumps', 'hiersea.csv')
MGSW_POST = get_matlab_resource(path.join('modelparams', 'mgsw_posterior.mat'),
                                variable_names=['beta_draws'])


class McmcTrace:
    """MCMC parameter traces"""
    def __init__(self, array):
        self._trace = np.array(array)

    def grab(self, param, foram):
        """Return array copy of MCMC trace parameter.
        """
        raise NotImplementedError


class PooledTrace(McmcTrace):
    """MCMC trace draws for pooled model.
    """
    def __init__(self, array):
        super().__init__(array)

    def grab(self, param):
        """Return array copy of MCMC trace parameter.
        """
        return self._trace[param].copy()


class HierTrace(McmcTrace):
    """MCMC trace draws for hierarchical model.
    """
    def __init__(self, array):
        super().__init__(array)
        self._forams = list(set([x.split('__')[-1] for x in self._trace.dtype.names if '__' in x]))

    @property
    def forams(self):
        return list(self._forams)

    def grab(self, param, foram=None):
        """Return array copy of MCMC trace parameter for a foraminifera.
        """
        if foram is None:
            return self._trace[param].copy()

        param_template = '{}__{}'
        try:
            return self._trace[param_template.format(param, foram)].copy()
        except ValueError:
            if any([x.find('{}__'.format(param)) != -1 for x in list(self._trace.dtype.names)]):
                # Likely bad foram name...
                msg_template = 'Bad `foram` arg: {}\nPossible `foram` are: {}'
                raise ForamError(msg_template.format(foram, self.forams))
            else:
                raise


class ForamError(Exception):
    def __init__(self, message):
        """Error raised if user passes bad foram str.
        """
        super().__init__(message)


class DrawDispenser:
    def __init__(self, pooled_annual=None, hier_annual=None, hier_seasonal=None):
        """Handles and passes out MCMC trace draws.

        Parameters
        ----------
        pooled_annual : PooledTrace
        hier_annual : HierTrace
        hier_seasonal : HierTrace
        """
        self.pooled_annual = pooled_annual
        self.hier_annual = hier_annual
        self.hier_seasonal = hier_seasonal

    def __call__(self, foram=None, seasonal_seatemp=False):
        """Get MCMC trace draws.
        """

        # For legacy, we're converting new foram species/subspecies to old,
        # short-hand names.
        foram_map = {'G. bulloides': 'bulloides',
                     'N. pachyderma sinistral': 'pachy_s',
                     'G. ruber pink': 'ruber_p',
                     'G. ruber white': 'ruber_w',
                     'G. sacculifer': 'sacculifer',
                     }
        new_name = foram_map.get(foram)
        if new_name is not None:
            foram = new_name

        if foram is None:
            trace = self.pooled_annual
            alpha = trace.grab('alpha')
            beta_temp = trace.grab('beta_temp')
            beta_omega = trace.grab('beta_omega')
            beta_clean = trace.grab('beta_clean')
            sigma = trace.grab('sigma')

        else:
            if seasonal_seatemp == True:
                trace = self.hier_seasonal
            else:
                trace = self.hier_annual

            alpha = trace.grab('alpha', foram)
            beta_temp = trace.grab('beta_temp', foram)
            beta_omega = trace.grab('beta_omega', foram)
            beta_clean = trace.grab('beta_clean')
            sigma = trace.grab('sigma', foram)

        return alpha, beta_temp, beta_omega, beta_clean, sigma


def get_sw_draws():
    """Return copy of arrays for Deep Time Mg/Ca seawater correction.
    """
    return np.array(MGSW_POST['beta_draws'][:, ::2][:, :1500])


# Preloading these resources so only need to load once on bayfox import.
# Drawing every 20 parameter to get 1500 draws.
get_draws = DrawDispenser(pooled_annual=PooledTrace(get_csv_resource(POOLEDANNTRACE_PATH)[::20]),
                          hier_annual=HierTrace(get_csv_resource(HIERANNTRACE_PATH)[::20]),
                          hier_seasonal=HierTrace(get_csv_resource(HIERSEASTRACE_PATH)[::20]))
