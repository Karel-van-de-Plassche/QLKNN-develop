import os
import time

import xarray as xr
import numpy as np

from qlknn.dataset.hypercube_to_pandas import *

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
    ds, ds_kwargs = open_with_disk_chunks(os.path.join(rootdir, 'Zeffcombo.nc.1'), dask=True)
    ds_sep = xr.open_dataset(os.path.join(rootdir, 'Zeffcombo.sep.nc.1'),
                             **ds_kwargs)
    ds_tot = ds.merge(ds_sep.data_vars)
    return ds_tot, ds_kwargs

@profile
def prep_megarun_ds(prepared_ds_name, starttime=None, rootdir='.', use_gam_cache=False, ds_loader=load_megarun1_ds):
    """ Prepares a QuaLiKiz netCDF4 dataset for convertion to pandas
    This function was designed to use dask, but should work for
    pure xarray too. In this function it is assumed the chunks on disk,
    and the chunks for dask are the same (or at least aligned)

    Kwargs:
        starttime:      Time the script was started. All debug timetraces will be
                        relative to this point. [Default: current time]
        rootdir:        Path where all un-prepared datasets reside [Default: '.']
        use_disk_cache: Just load an already prepared dataset [Default: False]
        use_gam_cache:  Load the already prepared gam_leq/gam_great cache [Default: False]
        ds_loader:      Function that loads all non-prepared datasets and merges
                        them to one. See default for example [Default: load_megarun1_ds]

    Returns:
        ds:             Prepared xarray.Dataset (chunked if ds_loader returned chunked)
    """
    if starttime is None:
        starttime = time.time()

    prepared_ds_path = os.path.join(rootdir, prepared_ds_name)
    # Load the dataset
    ds, ds_kwargs = ds_loader(rootdir)
    notify_task_done('Datasets merging', starttime)
    # Calculate gam_leq and gam_great and cache to disk
    ds = merge_gam_leq_great(ds,
                             ds_kwargs=ds_kwargs,
                             rootdir=rootdir,
                             use_disk_cache=use_gam_cache,
                             starttime=starttime)
    notify_task_done('gam_[leq,great]_GB cache creation', starttime)

    ds = determine_stability(ds)
    notify_task_done('[ITG|TEM] calculation', starttime)

    if 'normni' not in ds:
        ds = calculate_normni(ds)
    ds = absambi(ds)
    notify_task_done('absambi calculation', starttime)

    ds = calculate_particle_sepfluxes(ds)
    notify_task_done('pf[i|e][ITG|TEM] calculation', starttime)

    ds = sum_pinch(ds)
    notify_task_done('Total pinch calculation', starttime)
    # Remove variables and coordinates we do not need for NN training
    ds = ds.drop(['gam_GB', 'ome_GB'])
    # normni does not need to be saved for the 9D case.
    # TODO: Check for Aarons case!
    #if 'normni' in ds.data_vars:
    #    ds.attrs['normni'] = ds['normni']
    #    ds = ds.drop('normni')


    # Remove all but first ion
    # TODO: Check for Aarons case!
    ds = ds.sel(nions=0)
    ds.attrs['nions'] = ds['nions'].values
    ds = ds.drop('nions')
    ds = metadatize(ds)
    notify_task_done('Bookkeeping', starttime)

    notify_task_done('Pre-disk write dataset preparation', starttime)
    return ds, ds_kwargs

def prepare_megarun1(rootdir):
    starttime = time.time()
    store_name = os.path.join(rootdir, 'gen4_9D_nions0_flat_filter10.h5.1')
    prep_ds_name = 'Zeffcombo_prepared.nc.1'
    prep_ds_path = os.path.join(rootdir, prep_ds_name)
    ds_loader = load_megarun1_ds
    ds, ds_kwargs = prep_megarun_ds(prep_ds_name,
                         starttime=starttime,
                         rootdir=rootdir,
                         ds_loader=ds_loader)
    ds = remove_rotation(ds)
    save_prepared_ds(ds, prep_ds_path, starttime=starttime, ds_kwargs=None)
    notify_task_done('Preparing dataset', starttime)
    return ds, store_name

@profile
def load_rot_three_ds(rootdir='.'):
    ds, ds_kwargs = open_with_disk_chunks(os.path.join(rootdir, 'rot_three.nc.1'), dask=False)
    return ds, ds_kwargs

def prepare_rot_three(rootdir, use_disk_cache=False):
    starttime = time.time()
    store_name = os.path.join(rootdir, 'gen4_8D_rot_three.h5.1')
    prep_ds_name = 'rot_three_prepared.nc.1'
    prepared_ds_path = os.path.join(rootdir, prep_ds_name)
    ds_loader = load_rot_three_ds
    if use_disk_cache:
        ds, ds_kwargs = open_with_disk_chunks(prepared_ds_path, dask=False)
    else:
        ds, ds_kwargs = prep_megarun_ds(prep_ds_name,
                             starttime=starttime,
                             rootdir=rootdir,
                             ds_loader=ds_loader)

        # Drop SI variables
        for name, var in ds.variables.items():
            if name.endswith('_SI'):
                ds = ds.drop(name)

        # Remove ETG vars, rotation run is with kthetarhos <=2
        for name, var in ds.variables.items():
            if 'ETG' in name:
                print('Dropping {!s}'.format(name))
                ds = ds.drop(name)

        #ds = calculate_rotdivs(ds)
        save_prepared_ds(ds, prepared_ds_path, starttime=starttime, ds_kwargs=ds_kwargs)
    notify_task_done('Preparing dataset', starttime)
    return ds, store_name

@profile
def load_edge_one_ds(rootdir='.'):
    ds, ds_kwargs = open_with_disk_chunks(os.path.join(rootdir, 'Nustar0.35-1Ti_Te_rel1.0epsilon0.95.nc'), dask=False)
    return ds, ds_kwargs

def prepare_edge_one(rootdir, use_disk_cache=False):
    starttime = time.time()
    store_name = os.path.join(rootdir, 'gen4_6D_edge_one.h5.1')
    prep_ds_name = 'edge_one_prepared.nc.1'
    prepared_ds_path = os.path.join(rootdir, prep_ds_name)
    ds_loader = load_edge_one_ds
    if use_disk_cache:
        ds, ds_kwargs = open_with_disk_chunks(prepared_ds_path, dask=False)
    else:
        ds, ds_kwargs = prep_megarun_ds(prep_ds_name,
                             starttime=starttime,
                             rootdir=rootdir,
                             ds_loader=ds_loader)

        # Drop SI variables
        for name, var in ds.variables.items():
            if name.endswith('_SI'):
                ds = ds.drop(name)

        save_prepared_ds(ds, prepared_ds_path, starttime=starttime, ds_kwargs=ds_kwargs)
    notify_task_done('Preparing dataset', starttime)
    return ds, store_name

if __name__ == '__main__':
    #client = Client(processes=False)
    #client = Client()
    rootdir = '../../../qlk_data'
    ds, store_name = prepare_edge_one(rootdir)
    #ds, store_name = prepare_rot_three(rootdir)
    #ds, store_name = prepare_megarun1(rootdir)

    # Convert to pandas
    # Remove all variables with more dims than our cube
    non_drop_dims = list(ds[dummy_var].dims)
    for name, var in ds.items():
        if len(set(var.dims) - set(non_drop_dims)) != 0:
            ds = ds.drop(name)

    #dummy_var = next(ds.data_vars.keys().__iter__())
    ds['dimx'] = (ds[dummy_var].dims, np.arange(0, ds[dummy_var].size).reshape(ds[dummy_var].shape))
    use_disk_cache = False
    #use_disk_cache = True
    cachedir = os.path.join(rootdir, 'cache')
    if not use_disk_cache:
        create_input_cache(ds, cachedir)

    input_hdf5_from_cache(store_name, cachedir, columns=non_drop_dims, mode='w')
    save_attrs(ds.attrs, store_name)

    data_hdf5_from_ds(ds, store_name)
