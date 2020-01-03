"""Core functions to fetch modern seawater pH and omega values.
"""


__all__ = ['fetch_ph', 'fetch_omega']


import shapely.geometry
import xarray as xr
import numpy as np

from baymag.utils import get_matlab_resource, get_netcdf_resource


def chord_distance(latlon1, latlon2):
    """Chordal distance between two sequences of (lat, lon) points

    Parameters
    ----------
    latlon1 : sequence of tuples
        (latitude, longitude) for one set of points.
    latlon2 : sequence of tuples
        A sequence of (latitude, longitude) for another set of points.

    Returns
    -------
    dists : 2d array
        An mxn array of Earth chordal distances [1]_ (km) between points in
        latlon1 and latlon2.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Chord_(geometry)

    """
    earth_radius = 6378.137  # in km

    latlon1 = np.atleast_2d(latlon1)
    latlon2 = np.atleast_2d(latlon2)

    n = latlon1.shape[0]
    m = latlon2.shape[0]

    paired = np.hstack((np.kron(latlon1, np.ones((m, 1))),
                        np.kron(np.ones((n, 1)), latlon2)))

    latdif = np.deg2rad(paired[:, 0] - paired[:, 2])
    londif = np.deg2rad(paired[:, 1] - paired[:, 3])

    a = np.sin(latdif / 2) ** 2
    b = np.cos(np.deg2rad(paired[:, 0]))
    c = np.cos(np.deg2rad(paired[:, 2]))
    d = np.sin(np.abs(londif) / 2) ** 2

    half_angles = np.arcsin(np.sqrt(a + b * c * d))

    dists = 2 * earth_radius * np.sin(half_angles)

    return dists.reshape(m, n)


def fetch_ph(latlon):
    """Fetch modern seawater surface insitu pH.

    Parameters
    ----------
    latlon : tuple of floats
        Latitude and longitude of site. Latitude must be between -90 and 90.
        Longitude between -180 and 180.

    Returns
    -------
    ph : float
        In-situ pH.
    """
    # Very literal version of Jess Tierney's `omgph` for MATLAB. I'm so sorry.
    ph_ds = get_netcdf_resource('omgph/observations/GLODAPv2.2016b.pHtsinsitutp_subset.nc')[
        ['pHtsinsitutp', 'pHtsinsitutp_error', 'lat', 'lon']]
    
    lat_f = ph_ds.lat.values
    lon_f = ph_ds.lon.values
    ph_field = ph_ds.pHtsinsitutp.values.T

    nlon = len(lon_f)

    lon_f = np.concatenate([lon_f[(int(nlon / 2) - 20):(nlon - 20)] - 360,
                            lon_f[(nlon - 20):nlon] - 360,
                            lon_f[:(int(nlon / 2) - 20)]])
    ph_field = np.concatenate([ph_field[(int(nlon / 2) - 20):(nlon - 20), ...],
                               ph_field[(nlon - 20):nlon, ...],
                               ph_field[:(int(nlon / 2) - 20), ...]])

    # manually cut westernmost caribbean so sites to the west of it
    # get assigned appropriately.
    lat_carib = lat_f[102:108]
    ph_carib = ph_field[107, 102:108]

    a, b = np.meshgrid(lon_f, lat_f)
    c = np.concatenate((a.T, b.T), axis=1)
    locs = c.reshape((int(np.multiply(*c.shape) / 2), 2), order='F')

    n_lon = len(lon_f)
    n_lat = len(lat_f)

    ph_vec = ph_field.reshape((n_lon * n_lat), order='F')

    locs_obs_ph = locs[~np.isnan(ph_vec)]
    ph_obs = ph_vec[~np.isnan(ph_vec)]

    # Polygons - to check whether our site falls in these areas
    gulf_mexico = shapely.geometry.Polygon([(-96.5, 16.5), (-100.3, 30.5),
                                            (-82, 30.5), (-80.5, 23)])
    caribbean = shapely.geometry.Polygon([(-77.5, 8), (-90.8, 18.6),
                                          (-82.4, 22.9), (-61.5, 17.5),
                                          (-61.5, 8.8)])
    target_location = shapely.geometry.Point(latlon[::-1])
    max_dist = 700

    # Jess' loop to get data
    if gulf_mexico.contains(target_location):
        gom_d_mat = get_matlab_resource('omgph/observations/gom.mat')
        gom_d = xr.Dataset({'ph': (['depth'], gom_d_mat['ph'].ravel())},
                           coords={'depth': (['depth'], gom_d_mat['depth'].ravel())})
        ph = gom_d['ph'].sel(depth=0, method='nearest').values
    elif caribbean.contains(target_location):
        lat1 = np.argmin(np.abs(latlon[0] - lat_carib))
        ph = ph_carib[lat1]
    else:
        # Closest location
        dists = chord_distance(latlon, locs_obs_ph[:, ::-1])
        dmin = np.min(dists)
        imin = np.argmin(dists)
        ph = ph_obs[imin]
        # We don't return these but in case they're needed for debug:
        dists_ph = dmin
    return ph


def fetch_omega(latlon, depth):
    """Calculate modern in situ calcite saturation state (omega) for a location.

    Parameters
    ----------
    latlon : tuple of floats
        Latitude and longitude of site. Latitude must be between -90 and 90.
        Longitude between -180 and 180.
    depth : float
        Water depth (m). Larger positive values indicate greater depth.

    Returns
    -------
    omega : flomgphoat
      Calcite saturation state calculated at in situ temperature and pressure.
    """
    # Very literal version of Jess Tierney's `fetch_omega` for MATLAB. I'm so sorry.
    omg_d = get_netcdf_resource('omgph/observations/GLODAPv2.2016b.OmegaC_subset.nc')[['OmegaC', 'Depth']]
    lat_f = omg_d.lat.values
    lon_f = omg_d.lon.values
    wdepth = omg_d.Depth.values
    omega_field = omg_d.OmegaC.values.T

    nlon = len(lon_f)

    lon_f = np.concatenate([lon_f[(int(nlon / 2) - 20):(nlon - 20)] - 360,
                            lon_f[(nlon - 20):nlon] - 360,
                            lon_f[:(int(nlon / 2) - 20)]])
    omega_field = np.concatenate([omega_field[(int(nlon / 2) - 20):(nlon - 20), ...],
                                  omega_field[(nlon - 20):nlon, ...],
                                  omega_field[:(int(nlon / 2) - 20), ...]])

    # manually cute westernmost caribbean
    lat_carib = lat_f[102:108]
    omega_carib = omega_field[107, 102:108, :].squeeze()

    a, b = np.meshgrid(lon_f, lat_f)
    c = np.concatenate((a.T, b.T), axis=1)
    locs = c.reshape((int(np.multiply(*c.shape) / 2), 2), order='F')

    n_lon = len(lon_f)
    n_lat = len(lat_f)
    n_depth = len(wdepth)

    omega_vec = omega_field.reshape((n_lon * n_lat, n_depth), order='F')

    locs_obs_omega = []
    omega_obs = []
    for i in range(len(wdepth)):
        locs_obs_omega.append(locs[~np.isnan(omega_vec[:, i]), :])
        omega_obs.append(omega_vec[~np.isnan(omega_vec[:, i]), i])

    # Polygons - to check whether our site falls in these areas
    gulf_mexico = shapely.geometry.Polygon([(-96.5, 16.5), (-100.3, 30.5),
                                            (-82, 30.5), (-80.5, 23)])
    caribbean = shapely.geometry.Polygon([(-77.5, 8), (-90.8, 18.6),
                                          (-82.4, 22.9), (-61.5, 17.5),
                                          (-61.5, 8.8)])

    target_location = shapely.geometry.Point(latlon[::-1])

    max_dist = 700

    # Jess' loop to get data
    if gulf_mexico.contains(target_location):
        gom_d_mat = get_matlab_resource('omgph/observations/gom.mat')
        gom_d = xr.Dataset({'omega': (['depth'], gom_d_mat['omega'].ravel())},
                           coords={'depth': (['depth'], gom_d_mat['depth'].ravel())})
        omega = gom_d['omega'].sel(depth=depth, method='nearest').values
    elif caribbean.contains(target_location):
        d1 = np.argmin(np.abs(depth - wdepth))
        lat1 = np.argmin(np.abs(latlon[0] - lat_carib))
        omega = omega_carib[lat1, d1]
        # "in case got NaN for omega":
        while np.isnan(omega):
            d1 -= 1
            omega = omega_carib[lat1, d1]

        # We don't return these but in case they're needed for debug:
        dists_om = 0
    else:
        # Closest depth, not sure we need array but
        dnew = np.argmin(np.abs(depth - wdepth))
        # Closest location
        dists = chord_distance(latlon, locs_obs_omega[dnew][:, ::-1])
        dmin = np.min(dists)
        imin = np.argmin(dists)
        while dmin > max_dist:
            dnew -= 1
            dists = chord_distance([latlon], locs_obs_omega[dnew][:, ::-1])
            dmin = np.min(dists)
            imin = np.argmin(dists)

        # We don't return these but in case they're needed for debug:
        d_diff = depth - wdepth[dnew]
        dists_om = dmin
        omega_now = omega_obs[dnew]
        omega = omega_now[imin]

    return omega
