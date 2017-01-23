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


ds = xr.open_dataset('./Zeff_combined_pruned.nc')
ds = ds.drop(['numsols', 'nions'])
dimx = np.prod([x for x in ds.dims.values()])
panda = pd.read_hdf('rowwise.hdf5')
random = np.random.permutation(np.arange(len(panda)))
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
    panda.sample(numsamp)
    return (time.time() - start)

def get_panda_ic_npindex(numsamp, epoch=0):
    start = time.time()
    panda.iloc(random[epoch:epoch_numsamp])
    return (time.time() - start)

embed()
result = pd
for numsamp in [100, 1000, 10000]:
    for epoch in range(10):
        get_panda_ic_sample(numsamp, epoch)
        get_panda_ic_npindex(numsamp, epoch)
        





#data = df.read_hdf('test.hdf5', 'test')
embed()
