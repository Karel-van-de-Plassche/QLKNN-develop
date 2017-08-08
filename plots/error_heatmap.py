from IPython import embed
#import mega_nn
import numpy as np
import scipy.stats as stats
import pandas as pd
import os
import sys
networks_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../networks'))
NNDB_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../NNDB'))
sys.path.append(networks_path)
sys.path.append(NNDB_path)
from model import Network, NetworkJSON
from run_model import QuaLiKizNDNN

import matplotlib.pyplot as plt
from load_data import load_data

def zero_linregress(x, y):
    x_ = x[:,np.newaxis]
    a, _, _, _ = np.linalg.lstsq(x_, y)

    f = a * x
    y_bar = np.sum(y) / len(y)
    ss_res = np.sum(np.square(y - f))
    ss_tot = np.sum(np.square(y - y_bar))
    r_value = np.sqrt(1 - ss_res / ss_tot)
    return a, r_value

##############################################################################
# Load Data                                                                  #
##############################################################################
_, df, __ = load_data(61)

##############################################################################
# Filter dataset                                                             #
##############################################################################
df = df[df['target']<60]
#df = df[df['prediction']<10]
df = df[df['target']>=0]

df = df[df['target']>0.1]
#df = df[df['prediction']>0]

x = df['target']
y = df['prediction']
res = df['residuals']

fig = plt.figure()
ax = fig.add_subplot(111)
##############################################################################
# Plot heatmap                                                               #
##############################################################################
heatmap, xedges, yedges = np.histogram2d(x,y, bins=80)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

cax = ax.imshow(np.log10(heatmap.T), extent=extent, origin='lower', aspect='auto')
##############################################################################
# Plot regression                                                            #
##############################################################################
slope, r_value = zero_linregress(x, y)
intercept = 0
x_reg = np.linspace(x.min(), x.max(), 50)
ax.plot(x_reg, slope * x_reg + intercept, c='black')
ax.text(.8,.9, "$R^2 = " + str(np.round(r_value**2, 3)) + '$', fontsize=15, transform=ax.transAxes)
fig.colorbar(cax)

#import seaborn as sns
#sns.set(style="darkgrid", color_codes=True)

#g = sns.jointplot("target", "prediction", data=df, kind="reg",
#                  xlim=(0, 60), ylim=(0, 60), color="r", size=7)
#g = sns.lmplot("target", "prediction", data=df)

plt.show()
embed()
