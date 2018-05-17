import time
from itertools import product
import gc

import xarray as xr
import numpy as np
import pandas as pd
from IPython import embed

from qualikiz_tools.qualikiz_io.outputfiles import xarray_to_pandas

def metadatize(ds):
    """ Move all non-axis dims to metadata """
    scan_dims = [dim for dim in ds.dims if dim != 'kthetarhos' and dim != 'nions' and dim != 'numsols']
    metadata = {}
    for name in ds:
        if (all([dim not in scan_dims for dim in ds[name].dims]) and name != 'kthetarhos' and name != 'nions' and name !='numsols'):
            metadata[name] = ds[name].values
            ds = ds.drop(name)
    ds.attrs = metadata
    return ds

def absambi(ds):
    """ Calculate absambi; ambipolairity check for two ions"""
    # TODO: Generalize for >2 ions
    n0, n1 = [-(ds['Zi'].sel(nions=1) - ds['Zeff']) / ( ds['Zi'].sel(nions=0) * (ds['Zi'].sel(nions=0) - ds['Zi']
    .sel(nions=1))), (ds['Zi'].sel(nions=0) - ds['Zeff']) / ( ds['Zi'].sel(nions=1) * (ds['Zi'].sel(nions=0) - ds
    ['Zi'].sel(nions=1)))]
    n0 = xr.DataArray(n0, coords={'Zeff': n0['Zeff'], 'nions': 0}, name='n0')
    n1 = xr.DataArray(n1, coords={'Zeff': n1['Zeff'], 'nions': 1}, name='n1')
    ds['n'] = xr.concat([n0, n1], dim='nions')
    ds['absambi'] = np.abs(((ds['pfi_GB'] * ds['n'] * ds['Zi']).sum('nions') / ds['pfe_GB'])) 
    if (ds['absambi'].isnull() & (ds['pfe_GB'] != 0)).sum() == 0:
        ds['absambi'] = ds['absambi'].fillna(1)
    else:
        raise Exception
    ds = ds.drop('n')
    return ds

def calculate_grow_vars(ds):
    """ Calculate maxiumum growth-rate based variables """
    gam = ds['gam_GB']
    gam = gam.max(dim='numsols')
    gam_great = gam.where(gam.kthetarhos>2, drop=True)
    ds['gam_great_GB'] = gam_great.max('kthetarhos')
    gam_leq = gam.where(gam.kthetarhos<=2, drop=True)
    ds['gam_leq_GB'] = gam_leq.max('kthetarhos')
    return ds

def determine_stability(ds):
    """ Determine if a point is TEM or ITG unstable """
    ome = ds['ome_GB']
    ome = ome.where(ome.kthetarhos <= 2, drop=True)
    ion_unstable = (gam_leq != 0)
    ds['TEM'] = (ion_unstable & (ome > 0).any(dim='numsols')).any(dim='kthetarhos')
    ds['ITG'] = (ion_unstable & (ome < 0).any(dim='numsols')).any(dim='kthetarhos')
    return ds

def prep_totflux(ds):
    """ Prepare variables in 'totflux' dataset for NN training """
    ds = absambi(ds)

    ds = metadatize(ds)

    ds = calculate_grow_vars(ds)

    ds = determine_stability(ds)

    ds = ds.drop(['gam_GB', 'ome_GB'])
    return ds

def sum_pf(df=None, vt=None, vr=0, vc=None, An=None):
    """ Calculate particle flux from diffusivity and pinch"""
    pf = df * An + vt + vr + vc
    return pf

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

def prep_sepflux(ds):
    """ Prepare variables in 'sepflux' dataset for NN training """
    ds = metadatize(ds)
    ds = calculate_particle_sepfluxes(ds)
    return ds

def remove_rotation(ds):
    for value in ['vfiTEM_GB', 'vfiITG_GB', 'vriTEM_GB', 'vriITG_GB']:
        try:
            ds = ds.drop(value)
        except ValueError:
            print('{!s} already removed'.format(value))
    return ds

def load_totsepset():
    ds = xr.open_dataset('Zeffcombo.nc.1')
    ds = prep_totflux(ds)
    ds_sep = xr.open_dataset('Zeffcombo.sep.nc.1')
    ds_sep = prep_sepflux(ds_sep)
    ds_tot = ds.merge(ds_sep)
    return ds_tot

def prep_megarun1_ds(starttime=None):
    if starttime is None:
        starttime = time.time()

    ds_tot = load_totsepset()
    print('Datasets merged after', time.time() - starttime)

    ds_tot = remove_rotation(ds_tot)
    ds_tot.to_netcdf('Zeffcombo.combo.nc', format='NETCDF4', engine='netcdf4')
    print('Checkpoint ds_tot saved after', time.time() - starttime)

    # Remove all but first ion
    ds_tot = ds_tot.sel(nions=0)
    ds_tot.to_netcdf('Zeffcombo.combo.nions0.nc', format='NETCDF4', engine='netcdf4')
    return ds_tot

if __name__ == '__main__':
    starttime = time.time()
    ds_tot = prep_megarun1_ds(starttime=starttime)
    scan_dims = tuple(dim for dim in ds_tot.dims if dim != 'kthetarhos' and dim != 'nions' and dim != 'numsols')

    # Convert to pandas
    print('Starting pandaization after', time.time() - starttime)
    dfs = xarray_to_pandas(ds_tot)
    print('Xarray pandaized after', time.time() - starttime)
    del ds_tot
    gc.collect()

    store = pd.HDFStore('./gen3_9D_nions0_flat.h5')
    dfs[scan_dims].reset_index(inplace=True)
    dfs[scan_dims].index.name = 'dimx'
    print('Index reset after', time.time() - starttime)
    store['/megarun1/input'] = dfs[scan_dims].iloc[:, :len(scan_dims)]
    print('Input stored after', time.time() - starttime)
    store['/megarun1/flattened'] = dfs[scan_dims].iloc[:, len(scan_dims):]
    store['/megarun1/constants'] = dfs['constants']
    print('Done after', time.time() - starttime)
