from pkgutil import get_data
from io import BytesIO
from scipy.io import loadmat
import xarray as xr
import netCDF4

def get_matlab_resource(resource, package='mcforward', **kwargs):
    """Read flat MATLAB files as package resources, output for Numpy"""
    with BytesIO(get_data(package, resource)) as fl:
        data = loadmat(fl, **kwargs)
    return data


def get_netcdf_resource(resource, package='mcforward', **kwargs):
    """Read netCDF file as package resources, output xarray.Dataset"""
    with BytesIO(get_data(package, resource)) as fl:
        nc4_ds = netCDF4.Dataset(resource, memory=fl.read())
        store = xr.backends.NetCDF4DataStore(nc4_ds)
        data = xr.open_dataset(store, **kwargs)
    return data
