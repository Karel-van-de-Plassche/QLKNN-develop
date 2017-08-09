from IPython import embed
#import mega_nn
import numpy as np
import pandas as pd
from model import Network, NetworkJSON, TrainMetadata, Hyperparameters
from peewee import Param, JOIN_LEFT_OUTER
import os
import sys
networks_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../networks'))
sys.path.append(networks_path)
from run_model import QuaLiKizNDNN
import matplotlib as mpl
import matplotlib.pyplot as plt
from warnings import warn

#query = (Network.select(Network.id).where(Network.id == 16))
#nn = query.get().to_QuaLiKizNDNN()
def draw_convergence(network_id, only_last_epochs=False):
    query = (TrainMetadata.select(TrainMetadata.step,
                                  TrainMetadata.epoch,
                                  TrainMetadata.loss,
                                  TrainMetadata.walltime)
             .where(TrainMetadata.network == network_id)
             .dicts()
    )
    train = pd.DataFrame(query.where(TrainMetadata.set == 'train').get())
    #df.set_index('step', inplace=True)
    train.rename(columns = {'loss':'loss_train'}, inplace = True)
    train.index.name = 'minibatch'

    val = pd.DataFrame(query.where(TrainMetadata.set == 'validation').get())
    steps_per_epoch = val[val['epoch'] == 1]['step'].iloc[0]
    minibatches_per_epoch = steps_per_epoch - 1
    val.index = val['epoch'] * minibatches_per_epoch
    val.index.name = 'minibatch'
    val.rename(columns = {'loss':'loss_validation'}, inplace = True)
    #val.index = val.index - 1 # Shift steps to avond NaN gaps
    df = pd.concat([train, val], axis=1)
    if only_last_epochs:
        df = df.iloc[-100*steps_per_epoch-1:, :]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax1.scatter(df.index, df['loss_validation'], label='Validation')
    #for line in df['loss_validation'].dropna().index:
    #    ax1.axvline(line)
    ax1.scatter(df.index, df['loss_train'], s=4, label='Train')
    ax1.set_xticklabels(np.floor(ax1.xaxis.get_ticklocs()/minibatches_per_epoch))
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.legend()
    #ax1.set_ylim([-0.1, 20])

    def steps_to_walltime(steps):
        tottime = df['walltime'].iloc[-1, 1]
        times_seconds = steps * tottime / df.index[-1]
        return([seconds_to_human(x) for x in times_seconds])

    def seconds_to_human(c):
        days =    int(c // 86400)
        hours =   int(c // 3600 % 24)
        minutes = int(c // 60 % 60)
        seconds = int(c % 60)
        if days > 0:
            warn('Careful! Run took more than one day!')
        if c < 0:
            formatted = ''
        else:
            formatted = '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)
        return formatted

    ax2.set_xlim(ax1.get_xlim())
    #ax2.set_xticks(ax1.xaxis.get_ticklocs())
    ax2.set_xticklabels(steps_to_walltime(ax1.xaxis.get_ticklocs()))
    #ax2.set_xticklabels(ax1.xaxis.get_ticklocs())
    ax2.set_xlabel("time (HH:MM:SS)")


    hostname = (TrainMetadata.select(TrainMetadata.hostname)
             .where(TrainMetadata.network == network_id)
             .tuples()
    ).get()[0]
    ax2.text(0.01,0.98, hostname, fontsize=15, transform=fig.transFigure)
    ax2.text(0.01,0.96, 'network ' + str(network_id), fontsize=15, transform=fig.transFigure)
    return fig

def find_similar_convergence(network_id):
    query = Network.find_similar_topology_by_id(network_id)
    query &= Network.find_similar_networkpar_by_id(network_id)
    train_dim, minibatches, optimizer, standardization, hostname = (
        Network
        .select(Network.target_names,
                Hyperparameters.minibatches,
                Hyperparameters.optimizer,
                Hyperparameters.standardization,
                TrainMetadata.hostname)
        .where(Network.id == network_id)
        .join(Hyperparameters, on=Network.id == Hyperparameters.network_id)
        .join(TrainMetadata, on=Network.id == TrainMetadata.network_id)
    ).tuples().get()

    query &= (
        Network.select()
        .where(Network.target_names == Param(train_dim))
        .join(Hyperparameters)
        .where(Hyperparameters.minibatches == minibatches)
        #.where(Hyperparameters.optimizer == optimizer)
        .where(Hyperparameters.optimizer != 'adadelta')
        .where(Hyperparameters.standardization == standardization)
        .join(TrainMetadata, on=Network.id == TrainMetadata.network_id)
        .where(TrainMetadata.hostname == hostname)
    )
    query &= (
        Network.select()
        .where(Network.id != network_id)
    )

    if query.count() > 1:
        warn('multiple candidates! Returning the first')
    return query.get().id

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


orig_id = 46
draw_convergence(orig_id)
new_id = find_similar_convergence(orig_id)
draw_convergence(new_id)
#draw_mispred()
plt.show()
#embed()
