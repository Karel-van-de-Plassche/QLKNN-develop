import time
from itertools import product
import gc
import os

import xarray as xr
import numpy as np
import pandas as pd
from dask.distributed import Client
from dask.diagnostics import visualize
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
from IPython import embed

try:
    profile
except NameError:
    from qlknn.misc.tools import profile

from qualikiz_tools.qualikiz_io.outputfiles import xarray_to_pandas

GAM_LEQ_GB_TMP_PATH = 'gam_leq_GB.nc'

@profile
def metadatize(ds):
    """ Move all non-axis dims to metadata """
    scan_dims = [dim for dim in ds.dims if dim != 'kthetarhos' and dim != 'nions' and dim != 'numsols']
    metadata = {}
    for name in ds.coords:
        if (all([dim not in scan_dims for dim in ds[name].dims]) and name != 'kthetarhos' and name != 'nions' and name !='numsols'):
            metadata[name] = ds[name].values
            ds = ds.drop(name)
    ds.attrs = metadata
    return ds

@profile
def absambi(ds):
    """ Calculate absambi; ambipolairity check for two ions"""
    ds['absambi'] = (ds['pfi_GB'] * ds['normni'] * ds['Zi']).sum('nions') / ds['pfe_GB']
    ds['absambi'] = xr.where(ds['pfe_GB'] == 0, 1, ds['absambi'])
    return ds

@profile
def calculate_grow_vars(ds):
    """ Calculate maxiumum growth-rate based variables """

    kthetarhos = ds['kthetarhos']
    bound_idx = len(kthetarhos[kthetarhos <= 2])

    gam = ds['gam_GB']
    gam = gam.max('numsols')
    gam_leq = gam.isel(kthetarhos=slice(None, bound_idx))
    gam_leq = gam_leq.max('kthetarhos')
    gam_great = gam.isel(kthetarhos=slice(bound_idx + 1, None))
    gam_great = gam_great.max('kthetarhos')

    ds['gam_great_GB'] = gam_great
    ds['gam_leq_GB'] = gam_leq
    return ds

@profile
def determine_stability(ds):
    """ Determine if a point is TEM or ITG unstable """
    kthetarhos = ds['kthetarhos']
    bound_idx = len(kthetarhos[kthetarhos <= 2])
    ome = ds['ome_GB']

    gam_leq = ds['gam_leq_GB']
    ome = ome.isel(kthetarhos=slice(None, bound_idx))

    ion_unstable = (gam_leq != 0)
    embed()
    ds['TEM'] = ((ion_unstable & (ome > 0)).astype('bool').any(dim='numsols')).any(dim='kthetarhos')
    ds['ITG'] = ((ion_unstable & (ome < 0)).astype('bool').any(dim='numsols')).any(dim='kthetarhos')
    return ds

def prep_totflux(ds):
    """ Prepare variables in 'totflux' dataset for NN training """
    ds = absambi(ds)

    ds = metadatize(ds)

    #ds = calculate_grow_vars(ds)

    ds = determine_stability(ds)

    ds = ds.drop(['gam_GB', 'ome_GB'])
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
    and pinch files: df,vt,vc, and vr
    """
    for spec, mode in product(['e', 'i'], ['ITG', 'TEM']):
        fluxes = ['df', 'vt', 'vc']
        if spec == 'i':
            fluxes.append('vr')
        parts = {flux: flux + spec + mode + '_GB' for flux in fluxes}
        parts = {flux: ds[part] for flux, part in parts.items()}
        pf = sum_pf(**parts, An=ds['An'])
        pf.name = 'pf' + spec + mode + '_GB'
        ds[pf.name] = pf
    return ds

@profile
def prep_sepflux(ds):
    """ Prepare variables in 'sepflux' dataset for NN training """
    ds = metadatize(ds)
    ds = calculate_particle_sepfluxes(ds)
    return ds

@profile
def remove_rotation(ds):
    for value in ['vfiTEM_GB', 'vfiITG_GB', 'vriTEM_GB', 'vriITG_GB']:
        try:
            ds = ds.drop(value)
        except ValueError:
            print('{!s} already removed'.format(value))
    return ds

@profile
def load_megarun1_ds(rootdir='.'):
    # Determine the on-disk chunk sizes
    ds = xr.open_dataset(os.path.join(rootdir, 'Zeffcombo.nc.1'))
    chunk_sizes = ds['gam_GB']._variable._encoding['chunksizes']
    dims = ds['gam_GB']._variable.dims
    ds.close()

    # Re-open dataset with on-disk chunksizes
    ds_kwargs = {
        'chunks': dict(zip(dims, chunk_sizes)),
        #'cache': True
        #'lock': lock
              }
    ds = xr.open_dataset(os.path.join(rootdir, 'Zeffcombo.nc.1'),
                         **ds_kwargs)
    ds_sep = xr.open_dataset(os.path.join(rootdir, 'Zeffcombo.sep.nc.1'),
                             **ds_kwargs)
    ds_tot = ds.merge(ds_sep.data_vars)
    return ds_tot, ds_kwargs

@profile
def prep_megarun1_ds(starttime=None):
    if starttime is None:
        starttime = time.time()

    client = Client(processes=False)
    #client = Client()
    rootdir = '../../../qlk_data'
    ds, ds_kwargs = load_megarun1_ds(rootdir)
    print('Datasets merged after', time.time() - starttime)
    use_cached_gam_leq = True
    if not use_cached_gam_leq:
        ds = calculate_grow_vars(ds)
        ds_leq = ds['gam_leq_GB'].to_dataset()
        ds_leq = ds_leq.merge(ds.coords) # Save all coords
        ds_leq = profile(ds_leq.compute()) # Pre-compute, way faster as this fits in RAM!
        ds_leq.to_netcdf(os.path.join(rootdir, GAM_LEQ_GB_TMP_PATH))
    ds_leq = xr.open_dataset(os.path.join(rootdir, GAM_LEQ_GB_TMP_PATH),
                             **ds_kwargs)
    ds = ds.merge(ds, ds.data_vars)
    print('gam_leq_GB written after', time.time() - starttime)

    ds = prep_totflux(ds)
    print('Computed totflux after', time.time() - starttime)
    ds = prep_sepflux(ds)
    print('Computed sepflux after', time.time() - starttime)

    ds = remove_rotation(ds)
    #ds.to_netcdf('Zeffcombo.combo.nc', format='NETCDF4', engine='netcdf4')

    # Remove all but first ion
    ds = ds.sel(nions=0)

    print('starting after {:.0f}s'.format(time.time() - starttime))
    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
        ds.to_netcdf('Zeffcombo.combo.nions0.nc', format='NETCDF4', engine='netcdf4')
    print('prep_megarun1 done after {:.0f}s'.format(time.time() - starttime))
    visualize([prof, rprof, cprof], file_path='profile_' + backend + '.html')
    client.close()
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
    starttime = time.time()
    ds_tot = prep_megarun1_ds(starttime=starttime)
    scan_dims = tuple(dim for dim in ds_tot.dims if dim != 'kthetarhos' and dim != 'nions' and dim != 'numsols')

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
