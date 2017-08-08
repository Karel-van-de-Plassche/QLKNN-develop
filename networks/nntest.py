# coding: utf-8
from IPython import embed
import xarray as xr; from qualikiz_tools.qualikiz_io.outputfiles import *
import numpy as np; import pandas as pd; import matplotlib.pyplot as plt
from run_model import QuaLiKizNDNN
from simple_logging import load_hdf5

panda = load_hdf5('efe_GB.float16.h5')
train_dim = panda.columns[-1]
scan_dims = panda.columns[:-1]

root = os.path.dirname(os.path.realpath(__file__))
nn = QuaLiKizNDNN.from_json(os.path.join(root, 'nn.json'))
fluxes = nn.get_output(**panda[scan_dims])
fluxes.index = panda.index
panda[train_dim + 'NN'] = fluxes
print ('RMS = ' + str(np.sqrt(np.mean(np.square(panda[train_dim]
                                                - panda[train_dim + 'NN'])))))
embed()
