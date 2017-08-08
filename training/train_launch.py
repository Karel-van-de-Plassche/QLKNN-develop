import os
import sys
import json
from itertools import product
from IPython import embed
root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nns')
if not os.path.isdir(root):
    os.mkdir(root)

train_dim = 'efeETG_GB'
#l2_scales = [0.0, 0.05, 0.1, 0.2, 0.5, 1, 5, 10]
l2_scales = [0.1]
topologies = [[30, 30, 30], [60, 60], [120]]
filter = '3'

train_plan = {}
for l2_scale, topology in product(l2_scales, topologies):
    name = '_'.join([train_dim, str(l2_scale), str(topology)])
    train_plan[name] = {}
    with open('default_settings.json') as file_:
        settings = json.load(file_)
        settings['train_dim'] = train_dim
        settings['cost_l2_scale'] = l2_scale
        settings['hidden_neurons'] = topology
    train_plan[name]['settings'] = settings
    train_plan[name]['filter'] = settings
embed()
important_vars = ['train_dim', 'hidden_neurons', 'hidden_activation', 'output_activation']
['standardization', 'goodness', 'cost_l2_scale', 'cost_l1_scale']
['early_stop_after', 'optimizer']

def create_dir(name, settings):
    dir = os.path.join(root, name)
    os.mkdir(dir)

    os.symlink(os.path.abspath('train_NDNN.py'),
               os.path.join(dir, 'train_NDNN.py'))
    filtered_store_name = 'filtered_7D_nions0_flat_filter' + filter + '.h5'
    os.symlink(os.path.abspath(filtered_store_name),
               os.path.join(dir, 'filtered_everything_nions0.h5'))
    with open(os.path.join(dir, 'settings.json'), 'w') as file_:
        json.dump(settings, file_, indent=4)
