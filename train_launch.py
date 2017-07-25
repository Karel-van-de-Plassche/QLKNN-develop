import os
import sys
import json
root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nns')
if not os.path.isdir(root):
    os.mkdir(root)

train_dim = 'efeETG_GB'
l2_scales = [0.0, 0.05, 0.1, 0.2, 0.5, 1, 5, 10]
filter = '3'

for l2_scale in l2_scales:
    name = '_'.join([train_dim, str(l2_scale)])
    dir = os.path.join(root, name)
    os.mkdir(dir)

    os.symlink(os.path.abspath('train_NDNN.py'),
               os.path.join(dir, 'train_NDNN.py'))
    filtered_store_name = 'filtered_7D_nions0_flat_filter' + filter + '.h5'
    os.symlink(os.path.abspath(filtered_store_name),
               os.path.join(dir, 'filtered_everything_nions0.h5'))
    with open('default_settings.json') as file_:
        settings = json.load(file_)
        settings['train_dim'] = train_dim
        settings['cost_l2_scale'] = l2_scale
        with open(os.path.join(dir, 'settings.json'), 'w') as file_:
            json.dump(settings, file_, indent=4)
