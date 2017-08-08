from IPython import embed
#import mega_nn
import numpy as np
import pandas as pd
from model import Network, NetworkJSON, TrainMetadata
from peewee import Param
import os
import sys
networks_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../networks'))
sys.path.append(networks_path)
from run_model import QuaLiKizNDNN
import matplotlib as mpl
import matplotlib.pyplot as plt

#query = (Network.select(Network.id).where(Network.id == 16))
#nn = query.get().to_QuaLiKizNDNN()
def draw_convergence(network_id, only_last_epochs=True):
    query = (TrainMetadata.select(TrainMetadata.step,
                                  TrainMetadata.epoch,
                                  TrainMetadata.mse)
             .where(TrainMetadata.network == network_id)
             .dicts()
    )
    df = pd.DataFrame(query.where(TrainMetadata.set == 'train').get())
    df.set_index('step', inplace=True)
    df.rename(columns = {'mse':'mse_train'}, inplace = True)
    val = pd.DataFrame(query.where(TrainMetadata.set == 'validation').get())
    val.set_index('step', inplace=True)
    val.rename(columns = {'mse':'mse_validation'}, inplace = True)
    val.index = val.index - 1
    df = pd.concat([df, val], axis=1)
    if only_last_epochs:
        df = df.iloc[-100*10:, :]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df.index, np.sqrt(df['mse_validation']))
    ax.scatter(df.index, np.sqrt(df['mse_train']), s=4)
    ax.set_xticklabels(np.floor(ax.xaxis.get_ticklocs()/11))
    return fig

draw_convergence(22)
plt.show()
#embed()
