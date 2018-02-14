import xarray as xr
from IPython import embed
import numpy as np
from itertools import product
import pandas as pd
#import dask.dataframe as df
import time
#import dask.array as da
import numpy as np

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def timewrapper(func, *args, **kwargs):
    start = time.time()
    func(*args, **kwargs)
    print('Took ' + str(time.time() - start) + 's')

ds = xr.open_dataset('/global/cscratch1/sd/karel//Zeff_combined.nc')
ds = ds.drop([name for name, value in ds.data_vars.items() if 'kthetarhos' in value.dims])
ds = ds.drop([x for x in ds.coords if x not in ds.dims])
ds = ds.drop(['kthetarhos'])
ds = ds.max('numsols')
dimx = np.prod([x for x in ds.dims.values()])
traindim = 'efe_GB'
#fakeindex = cartesian(*[x for x in ds.dims.values()])
#panda = pd.read_hdf('/global/cscratch1/sd/karel/index.h5')
#random = np.random.permutation(np.arange(len(panda)))
#daarray = da.from_array(nparray, (10000, len(ds.dims)))
def iter_all(numsamp):
    start = time.time()
    cart = cartesian(ds.coords.values())
    nparray = np.empty((dimx, 9))
    for ii, foo in enumerate(product(*ds.coords.values())):
        nparray[ii, :] = list(map(float, foo))
        nparray[ii, :] = foo
        if ii > numsamp:
            break
    return (time.time() - start)

def get_panda_ic_sample(numsamp, epoch=0):
    start = time.time()
    set = panda.sample(numsamp)
    return (time.time() - start)

def get_panda_ic_npindex(numsamp, epoch=0):
    start = time.time()
    set = panda.iloc[random[epoch:(epoch + 1) * numsamp]]
    return (time.time() - start)


def get_panda_ic_npreindex(numsamp, epoch=0):
    start = time.time()
    idx = np.random.randint(0, dimx, numsamp)
    set = panda.iloc[idx]
    return (time.time() - start)


def get_xarray(numsamp, epoch=0):
    start = time.time()
    ds[traindim].isel_points(**{name: np.random.randint(0, len(value), numsamp) for name, value in ds['efe_GB'].coords.items()})
    return (time.time() - start)

strats = {
#    'panda_ic_sample': get_panda_ic_sample,
    'get_xarray': get_xarray,
#    'panda_ic_npindex': get_panda_ic_npindex,
#    'panda_ic_npreindex': get_panda_ic_npindex
}

numsamps = [1e3, 1e5, 1e6, 1e7, 1e8, dimx]
results = pd.DataFrame(columns=strats.keys(), index=[numsamps[0]])
numepochs = 3
embed()
for numsamp in numsamps:
    results.loc[numsamp] = None
    for name, func in strats.items():
        result = []
        for epoch in range(numepochs):
            result.append(func(numsamp, epoch))
            print(name, numsamp, str(epoch) + '/' + str(numepochs))
        print(name, numsamp, result[epoch])
        results[name].loc[numsamp] =  np.mean(result)
    print(results)
results.to_csv('benchmark_result.csv')
