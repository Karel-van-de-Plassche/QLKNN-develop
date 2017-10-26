import os
import sys
import json
from itertools import product
from IPython import embed
#root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nns')
#if not os.path.isdir(root):
#    os.mkdir(root)

def create_plan():
    train_dims = ['efeETG_GB']
    #l2_scales = [0.0, 0.05, 0.1, 0.2, 0.5, 1, 5, 10]
    l2_scales = [0.1]
    topologies = [[30, 30, 30], [60, 60], [120]]
    filter = 3
    dim = 7
    dataset_path = './filtered_{!s}D_nions0_flat_filter{!s}.h5'.format(dim, filter)
    train_plan = {}
    for l2_scale, topology in product(l2_scales, topologies):
        name = '_'.join(train_dims + [str(l2_scale), str(topology)])
        with open('training/default_settings.json') as file_:
            settings = json.load(file_)
            settings['train_dims'] = train_dims
            settings['cost_l2_scale'] = l2_scale
            settings['hidden_neurons'] = topology
            settings['dataset_path'] = dataset_path
        train_plan[name] = settings
        #train_plan[name]['filter'] = settings
    embed()
#    important_vars = ['train_dim', 'hidden_neurons', 'hidden_activation', 'output_activation']
#['standardization', 'goodness', 'cost_l2_scale', 'cost_l1_scale']
#['early_stop_after', 'optimizer']

def create_dir(name, settings):
    os.mkdir(name)

    os.symlink(os.path.abspath('train_NDNN.py'),
               os.path.join(name, 'train_NDNN.py'))

    with open(os.path.join(name, 'settings.json'), 'w') as file_:
        json.dump(settings, file_, indent=4)

def launch_training(path):
    old_dir = os.curdir()
    os.chdir(path)
