# coding: utf-8
from IPython import embed
import xarray as xr; from qualikiz_tools.qualikiz_io.outputfiles import *; import numpy as np; import pandas as pd; import matplotlib.pyplot as plt
from run_model import QuaLiKiz4DNN

ds = xr.open_dataset('/mnt/hdd/4D.nc')
df = ds.drop([coord for coord in ds.coords if coord not in ds.dims]).drop('kthetarhos').to_dataframe()
panda = pd.DataFrame(df.to_records())
scan_dims = [dim for dim in ds.dims if dim != 'kthetarhos' and dim != 'nions' and dim!='numsols']
train_dim = 'efe_GB'
panda = panda[scan_dims + [train_dim]]
panda = panda[panda[train_dim] > 0]
panda = panda[panda[train_dim] < 60]
nn = QuaLiKiz4DNN()
panda['est'] = nn.get_fluxes(*[value for name, value in (panda[scan_dims]).items()])
panda['est_scaled'] = nn.net['scale_factor']['efe_GB'] * nn.get_fluxes(*[value for name, value in (panda[scan_dims]).items()]) + nn.net['scale_bias']['efe_GB']
panda['scaled'] = nn.net['scale_factor']['efe_GB'] * panda['efe_GB'] + nn.net['scale_bias']['efe_GB']
np.mean(np.square(panda['scaled'] - panda['est']))
embed()
