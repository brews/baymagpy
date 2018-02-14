from pkgutil import get_data
from io import BytesIO
from co2sys import CO2SYS
from gsw.conversions import p_from_z

from mcforward.utils import get_matlab_resource, get_netcdf_resource


def get_omega(latlon, depth):
    """Calculate modern carbonate ion concentration and omega for a location

    Parameters
    ----------
    latlon : tuple of floats
        Latitude and longitude of site. Latitude must be between -90 and 90.
        Longitude between -180 and 180.
    depth : float
        Water depth (m).

    Returns
    -------
    out : float
    """
    raise NotImplementedError
    pres = p_from_z(z=-depth, lat=latlon[0])  # sea pressure ( i.e. absolute pressure - 10.1325 dbar )

    alk_d = get_netcdf_resource('omega/Alk.nc')
    dic_d = get_netcdf_resource('omega/TCO2.nc')
    # TODO(sbm): Looks like only the *_an var is read from these NC files. Maybe can reduce size?
    si_d = get_netcdf_resource('omega/woa13_Si_v2.nc', decode_times=False)[['i_an']]
    sal_d = get_netcdf_resource('omega/woa13_S_v2.nc', decode_times=False)[['s_an']]
    phos_d = get_netcdf_resource('omega/woa13_P_v2.nc', decode_times=False)[['p_an']]
    temp_d = get_netcdf_resource('omega/woa13_T_v2.nc', decode_times=False)[['t_an']]

    scs_d = get_matlab_resource('omega/scs.mat')

    med_alk_d = get_netcdf_resource('omega/med_alk.nc')
    ph_med_d = get_netcdf_resource('omega/med_ph.nc')
