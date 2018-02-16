import numpy as np


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


class DistanceThresholdError(Exception):
    """Raised when the distance between two points is further than a threshold

    Parameters
    ----------
        target_distance : int or float
            The distance between two target points (km).
        distance_threshold : int or float
            The distance threshold.

    """
    def __init__(self, target_distance, distance_threshold):
        self.target_distance = target_distance
        self.distance_threshold = distance_threshold


def get_nearest(latlon, dain, depth=None, lat_coord='lat', lon_coord='lon',
                depth_coord='depth', distance_threshold=1500):
    """Get nearest non-NaN to latlon from xarray.DataArray obj

    Finds the nearest not NaN to latlon, and optionally depth. It searches for a
    nearest depth first, if given, and then searches for nearest latlon. Note
    that this does not work with irregular grids, such as rotated polar, etc.

    Parameters
    ----------
    latlon : sequence
        Target latitude and longitude. Must be -90 to 90 and -180 to 180.
    dain : xarray.DataArray
        Field with regular latlon coordinates.
    depth : float or int, optional
        Target depth to get nearest.
    lat_coord : str, optional
        Name of the latitude coordinate in ``da``.
    lon_coord : str, optional
        Name of the longitude coordinate in ``da``.
    depth_coord : str, optional
        Name of the depth coordinate in ``da``.
    distance_threshold : float or int, optional
        If the nearest distance is larger than this, raise

    Returns
    -------
    nearest : xarray.DataArray
        Nearest points.
    nearest_distance : float
        Chordal distance (km) from target to matched gridpoint.

    Raises
    ------
    DistanceThresholdError
    """
    da = dain.copy()

    assert latlon[0] < 90 and latlon[0] > -90
    assert latlon[1] < 180 and latlon[1] >= -180

    assert lat_coord in da.coords
    assert lon_coord in da.coords

    assert (da[lat_coord].ndim == 1) and (da[lon_coord].ndim == 1)

    # First, find the nearest depth index, if given.
    if depth is not None:
        assert depth_coord in da.coords

        # Note use 'pad' because want next upper water column level value.
        da = da.sortby('depth')
        da = da.sel(**{depth_coord: depth}, method='pad')

    # Now search for nearest latlon point.
    da_stack = da.stack(yx=[lat_coord, lon_coord]).dropna('yx')
    da_latlon_stack = np.vstack((da_stack[lat_coord], da_stack[lon_coord])).T

    # Any values above 180 become negative -- needed for 0-360 longitudes.
    highlon_msk = da_latlon_stack > 180
    da_latlon_stack[highlon_msk] = da_latlon_stack[highlon_msk] - 360

    distance = chord_distance(np.array([latlon]), da_latlon_stack)
    nearest = da_stack.isel(yx=np.argmin(distance))
    nearest_distance = np.min(distance)

    if nearest_distance > distance_threshold:
        raise DistanceThresholdError(nearest_distance, distance_threshold)

    return nearest
