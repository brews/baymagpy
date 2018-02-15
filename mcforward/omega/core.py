from pkgutil import get_data
from io import BytesIO
from co2sys import CO2SYS
from gsw.conversions import p_from_z
import shapely.geometry

from mcforward.omega.utils import get_nearest
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
    assert latlon[0] < 90 and latlon[0] > -90
    assert latlon[1] < 180 and latlon[1] >= -180

    pres = p_from_z(z=-depth, lat=latlon[0])  # sea pressure ( i.e. absolute pressure - 10.1325 dbar )

    alk_d = get_netcdf_resource('omega/observations/Alk.nc')[['Alk']]
    dic_d = get_netcdf_resource('omega/observations/TCO2.nc')[['TCO2']]

    si_d = get_netcdf_resource('omega/observations/woa13_Si_v2.nc',
                               decode_times=False)[['i_an']]
    sal_d = get_netcdf_resource('omega/observations/woa13_S_v2.nc',
                                decode_times=False)[['s_an']]
    phos_d = get_netcdf_resource('omega/observations/woa13_P_v2.nc',
                                 decode_times=False)[['p_an']]
    temp_d = get_netcdf_resource('omega/observations/woa13_T_v2.nc',
                                 decode_times=False)[['t_an']]

    scs_d = get_matlab_resource('omega/observations/scs.mat')

    # Omegas from special places
    carib_d = get_matlab_resource('omega/observations/carib.mat')
    gom_d = get_matlab_resource('omega/observations/gom.mat')
    arctic_d = get_matlab_resource('omega/observations/arctic.mat')

    med_alk_d = get_netcdf_resource('omega/observations/med_alk.nc')[['a']]
    ph_med_d = get_netcdf_resource('omega/observations/med_ph.nc')[['a']]

    # set up polys for med, south china sea, caribbean, gulf of mexico.
    mediterranean = shapely.geometry.Polygon([(-5.5, 36.25),
                                              (3, 47.5),
                                              (45, 47.5),
                                              (45, 30),
                                              (-5.5, 30)])
    southchina_sea = shapely.geometry.Polygon([(106.2, 2.75),
                                          (104, 25),
                                          (119, 23),
                                          (120.5, 7)])
    caribbean = shapely.geometry.Polygon([(-77.5, 8),
                                          (-90.8, 18.6),
                                          (-82.4, 22.9),
                                          (-61.5, 17.5),
                                          (-61.5, 8.8)])
    gulf_mexico = shapely.geometry.Polygon([(-96.5, 16.5),
                                            (-100.3, 30.5),
                                            (-82, 30.5),
                                            (-80.5, 23)])
    # Remember we also have the arctic (lat > 65).

    target_location = shapely.geometry.Point(latlon[::-1])

    # Grab select variables from nearest gridpoints.
    # pH
    if mediterranean.contains(target_location):
        ph_s = get_nearest(latlon, ph_med_d['a'], depth=depth)

    # alk
    alk_s = get_nearest(latlon, alk_d['Alk'], depth=depth,
                        lat_coord='latitude', lon_coord='longitude')
    if southchina_sea.contains(target_location):
        raise NotImplementedError
        # TODO(brews): Finish S. China sea conditional - uses MATLAB file input. See ln 206-218 of grab_omega.m
        # alk_s = match
    elif mediterranean.contains(target_location):
        alk_s = get_nearest(latlon, med_alk_d['a'], depth=depth)

    # DIC
    dic_s = get_nearest(latlon, dic_d['TCO2'], depth=depth,
                        lat_coord='latitude', lon_coord='longitude')
    if southchina_sea.contains(target_location):
        raise NotImplementedError
        # TODO(brews): Finish S. China sea conditional - uses MATLAB file input. See ln 233-245 of grab_omega.m
        # dic_s = match

    # SI
    si_s = get_nearest(latlon, si_d['i_an'], depth=depth)

    # P
    p_s = get_nearest(latlon, phos_d['p_an'], depth=depth)

    # salinity
    sal_s = get_nearest(latlon, sal_d['s_an'], depth=depth)

    # Temperature
    temp_s = get_nearest(latlon, temp_d['t_an'], depth=depth)

    # Now plug all this into CO2SYS
    par1type = 1  # first param is "alkalinity"
    par2type = 2  # second param is "DIC"
    par3type = 3  # third param is "pH"
    presin = pres  # Pressure at input conditions
    tempout = 0  # Temperature at output conditions (doesn't matter)
    presout = 0  # Pressure at output conditions (also doesn't matter)
    phscale = 1  # pH scale of input pH vale - "Total scale" (doesn't matter)
    k1k2c = 4  # H2CO3 and HCO3- dissociation constants K1 and K2 - here "Mehrbach refit"
    kso4c = 1  # HSo4- dissociation constants KSo4 - "Dickson"

    omega = CO2SYS(alk_s, dic_s, par1type, par2type, sal_s, temp_s, tempout,
                   presin, presout, si_s, p_s, phscale, k1k2c, kso4c)[0]['OmegaCAin']
    if mediterranean.contains(target_location):
        omega = CO2SYS(alk_s, ph_s, par1type, par3type, sal_s, temp_s, tempout,
                       presin, presout, si_s, p_s, phscale, k1k2c, kso4c)[0]['OmegaCAin']
    elif caribbean.contains(target_location):
        # TODO(brews): Write this. Uses MATLAB file input. ln 314-321 of get_omega.m
        raise NotImplementedError
        # omega =
    elif gulf_mexico.contains(target_location):
        # TODO(brews): Write this. Uses MATLAB file input. ln 324-330 of get_omega.m
        raise NotImplementedError
    elif latlon[0] > 65:
        # TODO(brews): Write this. Uses MATLAB file input. ln 334-340 of get_omega.m
        raise NotImplementedError

    return omega
