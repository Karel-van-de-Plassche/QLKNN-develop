from IPython import embed
#import mega_nn
import numpy as np
import pandas as pd
from model import Network, NetworkJSON, TrainMetadata, Hyperparameters
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

def get_target_prediction(network_id):
    query = (Network.select(Network.target_names).where(Network.id==network_id).tuples())
    target_names = query[0][0]
    if len(target_names) == 1:
        target_name = target_names[0]
    else:
        NotImplementedError('Multiple targets not implemented yet')

    print(target_name)
    store = pd.HDFStore('./7D_nions0_flat.h5')
    input = store['megarun1/input']
    data = store['megarun1/flattened']
    try:
        se = data[target_name]
    except KeyError:
        raise Exception('Target name ' + str(target_name) + ' not found in dataset')
    try:
        root_name = '/megarun1/nndb_nn/'
        network_name = root_name + target_name + '/' + str(network_id)
        se_nn = store[network_name].iloc[:,0]
    except KeyError:
        raise Exception('Network name ' + network_name + ' not found in dataset')

    return se, se_nn

def calculate_zero_mispred(target, pred, threshold=0):
    zero_mispred = (target <= 0) & (pred > threshold)
    return np.sum(zero_mispred) / np.sum(target <= 0)

def calculate_zero_mispred_from_id(network_id, threshold=0):
    se, se_nn = get_target_prediction(network_id)
    mispred = calculate_zero_mispred(se, se_nn, threshold=threshold)
    return mispred

def draw_mispred():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lst = []
    for network_id in [61, 48, 46, 50, 49, 52, 53]:
        se, se_nn = get_target_prediction(network_id)
        query = (Network.select(Hyperparameters.cost_l2_scale).where(Network.id==network_id).join(Hyperparameters).tuples())
        l2_scale = query[0][0]
        for threshold in [0, 0.01, 0.1, 1]:
            mispred = calculate_zero_mispred(se, se_nn, threshold=threshold)
            lst.append({'threshold':threshold,
                          'l2_scale': l2_scale,
                          'mispred': mispred})
        continue
    df = pd.DataFrame(lst, columns=['threshold', 'l2_scale', 'mispred'])
    for threshold, frame in df.groupby('threshold'):
        ax.scatter(frame['l2_scale'], frame['mispred'], label=threshold)
    ax.set_xlabel('$c_{L2}$')
    ax.set_ylabel('misprediction rate [%]')
    plt.legend()


#draw_convergence(22)
draw_mispred()
plt.show()
#embed()
