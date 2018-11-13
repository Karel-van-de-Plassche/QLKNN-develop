import json
import os
import qlknn.training.train_NDNN as train_NDNN
from IPython import embed
from itertools import product
from collections import OrderedDict
from copy import deepcopy

root_dir = os.path.abspath('.')
tmproot = 'test'
if not os.path.exists(tmproot):
    os.makedirs(tmproot)

with open(os.path.join(root_dir, 'scan_settings.json')) as file_:
    scan = OrderedDict(json.load(file_))

with open(os.path.join(root_dir, 'settings.json')) as file_:
    base_settings = json.load(file_)

for ii, scan_tuple in enumerate(product(*scan.values())):
    settings = deepcopy(base_settings)
    for scan_var, val in zip(scan.keys(), scan_tuple):
        settings[scan_var] = val

    tmpdirname = os.path.join(tmproot, 'trainNN_' + str(ii))
    os.mkdir(tmpdirname)
    settings['dataset_path'] = os.path.abspath(settings['dataset_path'])
    with open(os.path.join(tmpdirname, 'settings.json'), 'w') as file_:
        json.dump(settings, file_, indent=4, sort_keys=True)
    os.chdir(tmpdirname)
    print('Starting job with {!s} = {!s}'.format(tuple(scan.keys()), scan_tuple))
    train_NDNN.train_NDNN_from_folder()
    os.chdir(root_dir)
