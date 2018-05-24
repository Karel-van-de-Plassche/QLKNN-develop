import os
import time

import numpy as np
from IPython import embed

def load_var(filepath, backend):
    if backend == 'h5py':
        import h5py
        f = h5py.File(filepath)
        var = f['gam_GB']
    elif backend == 'netcdf4':
        from dask.utils import SerializableLock
        HDF5_LOCK = SerializableLock()
        import netCDF4 as nc4
        mode = 'r'
        kwargs = {'format': 'NETCDF4',
                  'clobber': True,
                  'diskless': False,
                  'persist': False}
        ds = nc4.Dataset(filepath, mode=mode, **kwargs)
        var =  ds.variables['gam_GB']
    elif backend == 'xarray' or backend == 'xarray_dask':
        import xarray as xr
        ds = xr.open_dataset(filepath)
        var = ds['gam_GB']

    if backend == 'xarray_dask':
        dims, chunk_sizes = get_dims_chunks(var, 'xarray')
        chunks = dict(zip(dims, chunk_sizes))
        ds = xr.open_dataset(filepath, chunks=chunks)
        var = ds['gam_GB']
    return var

def get_dims_chunks(var, backend):
    if backend == 'h5py':
        dims = [var.file[dim[0]].name.lstrip('/')
                for dim in var.attrs['DIMENSION_LIST']]
        chunk_sizes = var.chunks
    elif backend == 'netcdf4':
        dims = var.dimensions
        chunk_sizes = var.chunking()
    elif backend == 'xarray_dask':
        dims = var.dims
        chunk_sizes = [chunk[0] if chunk[1:] == chunk[:-1] else None for chunk in var.chunks]
    elif backend == 'xarray':
        dims = var.dims
        chunk_sizes = var._variable._encoding['chunksizes']
    return dims, chunk_sizes

def process_var(var, backend, do_gam_great=False):
    if backend == 'h5py':
        kthetarhos = var.file['kthetarhos'].value
    elif backend == 'netcdf4':
        kthetarhos = var.group()['kthetarhos'][:].data
    elif backend in ['xarray', 'xarray_dask']:
        kthetarhos = var['kthetarhos'].values
    bound_idx = len(kthetarhos[kthetarhos <= 2])

    gam = var
    gam_great = None
    if backend in ['h5py', 'netcdf4']:
        import dask.array as da
        dims, chunk_sizes = get_dims_chunks(var, backend)

        gam = da.from_array(gam, chunks=chunk_sizes, lock=True)
        gam = gam.max(axis=dims.index('numsols'))
        gam_leq = gam[:,:,:,:,:,:,:,:,:, :bound_idx]
        gam_leq = gam_leq.max(axis=dims.index('kthetarhos'))
        if do_gam_great:
            gam_great = gam[:,:,:,:,:,:,:,:,:, bound_idx:]
            gam_great = gam_great.max(axis=dims.index('kthetarhos'))
    elif backend in ['xarray_dask', 'xarray']:
        #gam = ds['gam_GB'].chunk(chunks)
        starttime = time.time()
        gam = gam.max('numsols')
        gam_leq = gam.isel(kthetarhos=slice(None, bound_idx))
        gam_leq = gam_leq.max('kthetarhos')
        if do_gam_great:
            gam_great = gam.isel(kthetarhos=slice(bound_idx + 1, None))
            gam_great = gam_great.max('kthetarhos')
        if backend == 'xarray':
            print_done(backend, starttime)
    return gam_leq, gam_great

def print_done(backend, starttime):
    print('{!s} done after {:.0f}s'.format(backend, time.time() - starttime))

def doit(gam_leq, gam_great, backend, do_gam_great=False):
    from dask.distributed import Client
    from dask.diagnostics import visualize
    from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler

    client = Client(processes=False)

    print('starting {!s}'.format(backend))
    starttime = time.time()
    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
        gam_leq_out = gam_leq.compute(scheduler='threads')
        if do_gam_great:
            gam_great_out = gam_great.compute(scheduler='threads')
        else:
            gam_great_out = None

    visualize([prof, rprof, cprof], file_path='profile_' + backend + '.html')
    print_done(backend, starttime)
    client.close()
    return gam_leq_out, gam_great_out

file_dir = '../../../qlk_data'
filepath = os.path.join(file_dir, 'Zeffcombo_rerechunk.nc.1')
backends = ['h5py', 'netcdf4', 'xarray_dask', 'xarray']
#backend = 'h5py'
#backend = 'netcdf4'
#backend = 'xarray_dask'
for backend in backends:
    dsets = load_var(filepath, backend)
    try:
        gam_leq, gam_great = process_var(dsets, backend)
        doit(gam_leq, gam_great, backend)
    except MemoryError:
        print('Not enough memory to use backend {!s}'.format(backend))
