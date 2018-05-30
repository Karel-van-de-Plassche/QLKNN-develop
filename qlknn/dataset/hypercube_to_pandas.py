import time
from itertools import product
import gc
import os
import copy

import xarray as xr
import pandas as pd
from dask.distributed import Client, get_client
from dask.diagnostics import visualize
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
from dask.distributed import fire_and_forget
from IPython import embed

try:
    profile
except NameError:
    from qlknn.misc.tools import profile

from qualikiz_tools.qualikiz_io.outputfiles import xarray_to_pandas
from qlknn.misc.tools import notify_task_done

GAM_LEQ_GB_TMP_PATH = 'gam_cache.nc'

@profile
def metadatize(ds):
    """ Move all non-axis dims to metadata """
    scan_dims = [dim for dim in ds.dims
                 if dim not in ['kthetarhos', 'nions', 'numsols']]
    metadata = {}
    for name in ds.coords:
        if (all([dim not in scan_dims for dim in ds[name].dims]) and
                name not in ['kthetarhos', 'nions', 'numsols']):
            metadata[name] = ds[name].values
            ds = ds.drop(name)
    ds.attrs = metadata
    return ds

@profile
def absambi(ds):
    """ Calculate absambi; ambipolairity check (d(charge)/dt ~ 0)

    Args:
        ds:    Dataset containing pf[i|e]_GB, normni and Zi

    Returns:
        ds:    ds with absambi: sum(pfi * normni * Zi) / pfe
    """
    ds['absambi'] = ((ds['pfi_GB'] * ds['normni'] * ds['Zi']).sum('nions')
                     / ds['pfe_GB'])
    ds['absambi'] = xr.where(ds['pfe_GB'] == 0, 1, ds['absambi'])
    return ds

@profile
def determine_stability(ds):
    """ Determine if a point is TEM or ITG unstable. True if unstable """
    kthetarhos = ds['kthetarhos']
    bound_idx = len(kthetarhos[kthetarhos <= 2])
    ome = ds['ome_GB']

    gam_leq = ds['gam_leq_GB']
    ome = ome.isel(kthetarhos=slice(None, bound_idx))

    ion_unstable = (gam_leq != 0)
    ds['TEM'] = ion_unstable & (ome > 0).any(dim=['kthetarhos', 'numsols'])
    ds['ITG'] = ion_unstable & (ome < 0).any(dim=['kthetarhos', 'numsols'])
    return ds


@profile
def sum_pf(df=None, vt=None, vr=0, vc=None, An=None):
    """ Calculate particle flux from diffusivity and pinch"""
    pf = df * An + vt + vr + vc
    return pf

@profile
def calculate_particle_sepfluxes(ds):
    """ Calculate pf[i|e][ITG|TEM] from diffusivity and pinch

    This is needed because of a bug in QuaLiKiz 2.4.0 in which
    pf[i|e][ITG|TEM] was not written to file. Fortunately,
    this can be re-calculated from the saved diffusivity
    and pinch files: df,vt,vc, and vr. Will not do anything
    if already contained in ds.
    """
    for spec, mode in product(['e', 'i'], ['ITG', 'TEM']):
        pf_name = 'pf' + spec + mode + '_GB'
        if pf_name not in ds:
            fluxes = ['df', 'vt', 'vc']
            if spec == 'i':
                fluxes.append('vr')
            parts = {flux: flux + spec + mode + '_GB' for flux in fluxes}
            parts = {flux: ds[part] for flux, part in parts.items()}
            pf = sum_pf(**parts, An=ds['An'])
            pf.name = pf_name
            ds[pf.name] = pf
    return ds


@profile
def remove_rotation(ds):
    """ Drop the rotation-related variables from dataset """
    for value in ['vfiTEM_GB', 'vfiITG_GB', 'vriTEM_GB', 'vriITG_GB']:
        try:
            ds = ds.drop(value)
        except ValueError:
            print('{!s} already removed'.format(value))
    return ds

def open_with_disk_chunks(path):
    # Determine the on-disk chunk sizes
    ds = xr.open_dataset(path)
    if 'gam_GB' in ds.data_vars:
        chunk_sizes = ds['gam_GB']._variable._encoding['chunksizes']
        dims = ds['gam_GB']._variable.dims
    elif 'efe_GB' in ds.data_vars:
        chunk_sizes = ds['efe_GB']._variable._encoding['chunksizes']
        dims = ds['efe_GB']._variable.dims
    else:
        raise Exception('Could not figure out chunk sizes, no gam_GB nor efe_GB')
    ds.close()

    # Re-open dataset with on-disk chunksizes
    chunks = dict(zip(dims, chunk_sizes))
    ds_kwargs = {
        'chunks': chunks,
        #'cache': True
        #'lock': lock
    }
    ds = xr.open_dataset(path,
                         **ds_kwargs)
    return ds, ds_kwargs

@profile
def load_megarun1_ds(rootdir='.'):
    """ Load the 'megarun1' data as xarray/dask dataset
    For the megarun1 dataset, the data is split in the 'total fluxes + growth rates'
    and 'TEM/ITG/ETG fluxes'. Open the former with `open_with_disk_chunks` and the
    latter with the same kwargs and merge them together.

    Kwargs:
        starttime: Time the script was started. All debug timetraces will be
                   relative to this point. [Default: current time]

    Returns:
        ds:        Merged chunked xarray.Dataset ready for preparation.
    """
    ds, ds_kwargs = open_with_disk_chunks(os.path.join(rootdir, 'Zeffcombo.nc.1'))
    ds_sep = xr.open_dataset(os.path.join(rootdir, 'Zeffcombo.sep.nc.1'),
                             **ds_kwargs)
    ds_tot = ds.merge(ds_sep.data_vars)
    return ds_tot, ds_kwargs

def get_dims_chunks(var):
    if var.chunks is not None:
        if isinstance(var.chunks, dict):
            # xarray-style
            sizes = var.chunks
            chunksizes = [sizes[dim][0] if sizes[dim][:-1] == sizes[dim][1:] else None
                          for dim in var.dims]
        if isinstance(var.chunks, tuple):
            # dask-style
            chunksizes = [sizes[0] if sizes[1:] == sizes[:-1] else None for sizes in var.chunks]
        if None in chunksizes:
            raise Exception('Unequal size for one of the chunks in {!s}'.format(var.chunks.items()))
    else:
        raise NotImplementedError('Getting chunks of {!s}'.format(var))
    return chunksizes

@profile
def calculate_grow_vars(ds):
    """ Calculate maxiumum growth-rate based variables """

    kthetarhos = ds['kthetarhos']
    bound_idx = len(kthetarhos[kthetarhos <= 2])

    gam = ds['gam_GB']
    gam = gam.max('numsols')

    gam_leq = gam.isel(kthetarhos=slice(None, bound_idx))
    gam_leq = gam_leq.max('kthetarhos')
    gam_leq.name = 'gam_leq_GB'

    gam_great = gam.isel(kthetarhos=slice(bound_idx + 1, None))
    gam_great = gam_great.max('kthetarhos')
    gam_great.name = 'gam_great_GB'

    return gam_leq, gam_great

@profile
def merge_gam_leq_great(ds, ds_kwargs=None, rootdir='.', use_disk_cache=False, starttime=None):
    """ Calculate and cache (or load from cache) gam_leq and gam_great

    As advised by xarray http://xarray.pydata.org/en/stable/dask.html#optimization-tips
    save gam_leq (which is an intermediate result) to disk.

    Args:
        ds:             xarray.Dataset containing gam_GB

    Kwargs:
        ds_kwargs:      xarray.Dataset kwargs passed to xr.open_dataset. [Default: None]
        rootdir:        Directory where the cache will be saved/is saved
        use_disk_cache: Just load an already cached dataset [Default: False]
        starttime:      Time the script was started. All debug timetraces will be
                        relative to this point. [Default: current time]

    Returns:
        ds:             The xarray.Dataset with gam_leq and gam_great merged in
    """
    if ds_kwargs is None:
        ds_kwargs = {}

    if starttime is None:
        starttime = time.time()

    gam_cache_dir = os.path.join(rootdir, GAM_LEQ_GB_TMP_PATH)
    if not use_disk_cache:
        gam_leq, gam_great = calculate_grow_vars(ds)
        chunksizes = get_dims_chunks(gam_leq)

        # Perserve chunks/encoding from gam_GB
        encoding = {}
        for key in ['zlib', 'shuffle', 'complevel', 'dtype', ]:
            encoding[key] = ds['gam_GB'].encoding[key]
        encoding['chunksizes'] = chunksizes

        # We assume this fits in RAM, so load before writing to get some extra speed
        gam_leq.load()
        gam_leq.to_netcdf(gam_cache_dir, encoding={gam_leq.name: encoding})
        gam_great.load()
        gam_great.to_netcdf(gam_cache_dir, encoding={gam_great.name: encoding}, mode='a')

    # Now open the cache with the same args as the original dataset, as we
    # aggregated over kthetarhos and numsols, remove them from the chunk list
    kwargs = copy.deepcopy(ds_kwargs)
    chunks = kwargs.pop('chunks')
    for key in ['kthetarhos', 'numsols']:
        chunks.pop(key)

    # Finally, open and merge the cache
    ds_gam = xr.open_dataset(os.path.join(rootdir, GAM_LEQ_GB_TMP_PATH),
                             chunks=chunks,
                             **kwargs)
    ds = ds.merge(ds_gam.data_vars)
    return ds

def compute_and_save_var(ds, new_ds_path, varname, chunks, starttime=None):
    if starttime is None:
        starttime = time.time()

    print('starting', varname)
    #var = ds[varname]
    encoding = {varname: {'chunksizes':  [chunks[dim] for dim in ds[varname].dims], 'zlib': True}}
    #var.load()
    ds[varname].to_netcdf(new_ds_path, 'a',
                  encoding=encoding)
    notify_task_done(varname + ' saved', starttime)

@profile
def compute_and_save(ds, new_ds_path, chunks=None, starttime=None):
    """ Sequentially load all data_vars in RAM and write to new dataset
    This function forcibly loads variables in RAM, also triggering any
    lazy-loaded dask computations. We thus assume the result of each of
    these calculations fits in RAM. This is done because as of xarray
    '0.10.4', aligning on-disk chunks and dask chunks is still work in
    progress.

    Args:
        ds:          Dataset to write to new dataset
        new_ds_path: Absolute path to of the netCDF4 file that will
                     be generated

    Kwargs:
        chunks:      Dict with the dimension: chunksize for the new ds file
        starttime:   Time the script was started. All debug timetraces will be
                     relative to this point. [Default: current time]
    """
    if starttime is None:
        starttime = time.time()

    new_ds = xr.Dataset()
    for coord in ds.coords:
        new_ds.coords[coord] = ds[coord]
    new_ds.to_netcdf(new_ds_path)
    notify_task_done('Coords saving', starttime)
    print(chunks)

    data_vars = list(ds.data_vars)
    calced_dims = ['gam_leq_GB', 'gam_great_GB', 'TEM', 'ITG', 'absambi']
    for spec, mode in product(['e', 'i'], ['ITG', 'TEM']):
        flux = 'pf'
        calced_dims.append(flux + spec + mode + '_GB')

    for calced_dim in reversed(calced_dims):
        if calced_dim in ds:
            data_vars.insert(0, data_vars.pop(data_vars.index(calced_dim)))

    for varname in data_vars:
        #fire_and_forget(compute_and_save_var(ds, new_ds_path, varname, chunks, starttime=starttime))
        compute_and_save_var(ds, new_ds_path, varname, chunks, starttime=starttime)

@profile
def prep_megarun_ds(starttime=None, rootdir='.', use_disk_cache=False, ds_loader=load_megarun1_ds):
    """ Prepares a QuaLiKiz netCDF4 dataset for convertion to pandas
    This function was designed to use dask, but should work for
    pure xarray too. In this function it is assumed the chunks on disk,
    and the chunks for dask are the same (or at least aligned)

    Kwargs:
        starttime:      Time the script was started. All debug timetraces will be
                        relative to this point. [Default: current time]
        rootdir:        Path where all un-prepared datasets reside [Default: '.']
        use_disk_cache: Just load an already prepared dataset [Default: False]
        ds_loader:      Function that loads all non-prepared datasets and merges
                        them to one. See default for example [Default: load_megarun1_ds]

    Returns:
        ds:             Prepared xarray.Dataset (chunked if ds_loader returned chunked)
    """
    if starttime is None:
        starttime = time.time()

    prepared_ds_path = os.path.join(rootdir, 'Zeffcombo_prepared.nc.1')
    if not use_disk_cache:
        # Load the dataset
        ds, ds_kwargs = ds_loader(rootdir)
        notify_task_done('Datasets merging', starttime)
        use_disk_cache = True
        use_disk_cache = False
        # Calculate gam_leq and gam_great and cache to disk
        ds = merge_gam_leq_great(ds,
                                 ds_kwargs=ds_kwargs,
                                 rootdir=rootdir,
                                 use_disk_cache=use_disk_cache,
                                 starttime=starttime)
        notify_task_done('gam_[leq,great]_GB cache creation', starttime)

        ds = determine_stability(ds)
        notify_task_done('[ITG|TEM] calculation', starttime)

        ds = absambi(ds)
        notify_task_done('absambi calculation', starttime)

        ds = calculate_particle_sepfluxes(ds)
        notify_task_done('pf[i|e][ITG|TEM] calculation', starttime)
        # Remove variables and coordinates we do not need for NN training
        ds = ds.drop(['gam_GB', 'ome_GB'])
        ds = remove_rotation(ds)
        ds = metadatize(ds)
        # normni does not need to be saved for the 9D case.
        # TODO: Check for Aarons case!
        if 'normni' in ds.data_vars:
            ds.attrs['normni'] = ds['normni']
            ds = ds.drop('normni')

        notify_task_done('Bookkeeping', starttime)

        # Remove all but first ion
        ds = ds.sel(nions=0)

        # Save prepared dataset to disk
        notify_task_done('Pre-disk write dataset preparation', starttime)
        with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
            compute_and_save(ds, prepared_ds_path, chunks=ds_kwargs['chunks'], starttime=starttime)
        notify_task_done('prep_megarun', starttime)
        visualize([prof, rprof, cprof], file_path='profile_prep.html')
    else:
        ds, __ = open_with_disk_chunks(prepared_ds_path)
    return ds

@profile
def extract_trainframe(dfs):
    store = pd.HDFStore('./gen3_9D_nions0_flat.h5')
    dfs[scan_dims].reset_index(inplace=True)
    dfs[scan_dims].index.name = 'dimx'
    return dfs[scan_dims], dfs['constants']

@profile
def save_trainframe(df, constants):
    store['/megarun1/input'] = df.iloc[:, :len(scan_dims)]
    print('Input stored after', time.time() - starttime)
    store['/megarun1/flattened'] = df.iloc[:, len(scan_dims):]
    store['/megarun1/constants'] = constants

if __name__ == '__main__':
    client = Client(processes=False)
    starttime = time.time()
    rootdir = '../../../qlk_data'
    use_disk_cache = True
    use_disk_cache = False
    ds = prep_megarun_ds(starttime=starttime,
                         rootdir=rootdir,
                         use_disk_cache=use_disk_cache,
                         ds_loader=load_megarun1_ds)
    client.close()
    print('Done')
    embed()
    exit()
    #scan_dims = tuple(dim for dim in ds_tot.dims if dim != 'kthetarhos' and dim != 'nions' and dim != 'numsols')

    # Convert to pandas
    print('Starting pandaization after', time.time() - starttime)
    dfs = profile(xarray_to_pandas(ds_tot))
    print('Xarray pandaized after', time.time() - starttime)
    del ds_tot
    gc.collect()
    df, constants = extract_trainframe(dfs)
    print('Trainframe extracted after', time.time() - starttime)
    save_trainframe(df, constants)
    print('Done after', time.time() - starttime)
