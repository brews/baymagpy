from co2sys import CO2SYS
from gsw.conversions import p_from_z
import shapely.geometry
import xarray as xr

from mcforward.omega.utils import get_nearest, DistanceThresholdError
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

    # South China Sea Data
    scs_d_mat = get_matlab_resource('omega/observations/scs.mat')
    scs_d = xr.Dataset({'alk': (['depth'], scs_d_mat['alk'].ravel()),
                        'tco2': (['depth'], scs_d_mat['tco2'].ravel())},
                       coords={'depth': (['depth'], scs_d_mat['depth'].ravel())})

    # Caribbean
    carib_d_mat = get_matlab_resource('omega/observations/carib.mat')
    carib_d = xr.Dataset({'ph': (['depth'], carib_d_mat['ph'].ravel()),
                          'omega': (['depth'], carib_d_mat['omega'].ravel()),
                          'carb': (['depth'], carib_d_mat['carb'].ravel())},
                         coords={'depth': (['depth'], carib_d_mat['depth'].ravel())})

    # Gulf of Mexico
    gom_d_mat = get_matlab_resource('omega/observations/gom.mat')
    gom_d = xr.Dataset({'ph': (['depth'], gom_d_mat['ph'].ravel()),
                        'omega': (['depth'], gom_d_mat['omega'].ravel()),
                        'carb': (['depth'], gom_d_mat['carb'].ravel())},
                       coords={'depth': (['depth'], gom_d_mat['depth'].ravel())})

    # Arctic
    arctic_d_mat = get_matlab_resource('omega/observations/arctic.mat')
    arctic_d = xr.Dataset({'ph': (['depth'], arctic_d_mat['ph'].ravel()),
                           'omega': (['depth'], arctic_d_mat['omega'].ravel()),
                           'carb': (['depth'], arctic_d_mat['carb'].ravel())},
                          coords={'depth': (['depth'], arctic_d_mat['depth'].ravel())})

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
    if southchina_sea.contains(target_location):
        alk_s = scs_d['alk'].sel(depth=depth, method='pad')
    elif mediterranean.contains(target_location):
        alk_s = get_nearest(latlon, med_alk_d['a'], depth=depth)
    else:
        try:
            alk_s = get_nearest(latlon, alk_d['Alk'], depth=depth,
                                lat_coord='latitude', lon_coord='longitude')
        except DistanceThresholdError:
            alk_s = None

    # DIC
    if southchina_sea.contains(target_location):
        dic_s = scs_d['tco2'].sel(depth=depth, method='pad')
    else:
        try:
            dic_s = get_nearest(latlon, dic_d['TCO2'], depth=depth,
                                lat_coord='latitude', lon_coord='longitude')
        except DistanceThresholdError:
            dic_s = None

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

    if mediterranean.contains(target_location):
        omega = CO2SYS(alk_s, ph_s, par1type, par3type, sal_s, temp_s, tempout,
                       presin, presout, si_s, p_s, phscale, k1k2c, kso4c)[0]['OmegaCAin']
    elif caribbean.contains(target_location):
        omega = carib_d['omega'].sel(depth=depth, method='pad').values
    elif gulf_mexico.contains(target_location):
        omega = gom_d['omega'].sel(depth=depth, method='pad').values
    elif latlon[0] > 65:
        omega = arctic_d['omega'].sel(depth=depth, method='pad').values
    else:
        omega = CO2SYS(alk_s, dic_s, par1type, par2type, sal_s, temp_s, tempout,
                       presin, presout, si_s, p_s, phscale, k1k2c, kso4c)[0]['OmegaCAin']

    return omega
