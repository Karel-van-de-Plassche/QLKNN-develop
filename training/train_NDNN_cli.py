#!/usr/bin/env python3
"""
qualikiz_tools

Usage:
  train_NDNN_cli [-v | -vv] [options] <command>

Options:
  -h --help                         Show this screen.
  --train_dims=<train_dims>                      Training dimensions [default: '']
  --dataset_path=<dataset_path>                    [default: filtered_everything_nions0.h5]
  --drop_outlier_above=<drop_outlier_above>              [default: 1.]
  --hidden_neurons=<hidden_neurons>                  [default: 30,30,30]
  --hidden_activation=<hidden_activation>               [default: 'tanh','tanh','tanh']
  --drop_chance=<drop_chance>                     [default: 0.0]
  --output_activation=<output_activation>         [default: none]
  --standardization=<standardization>                 [default: normsm_1_0]
  --goodness=<goodness>                        [default: mse]
  --cost_l2_scale=<cost_l2_scale>                   [default: 0.1]
  --cost_l1_scale=<cost_l1_scale>                   [default: 0.0]
  --early_stop_after=<early_stop_after>                [default: 5]
  --early_stop_measure=<early_stop_measure>              [default: loss]
  --minibatches=<minibatches>                     [default: 10]

  --validation_fraction=<validation_fraction>             [default: 0.05]
  --test_fraction=<test_fraction>                   [default: 0.05]
  --dtype=<dtype>                           [default: float32]

  --optimizer=<optimizer>                       [default: adam]
  --learning_rate=<learning_rate>           [default: 0.001]

  --lbfgs_maxfun=<lbfgs_maxfun>                    [default: 1000]
  --lbfgs_maxiter=<lbfgs_maxiter>                   [default: 15000]
  --lbfgs_maxls=<lbfgs_maxls>                     [default: 20]
  --adam_beta1=<adam_beta1>                      [default: 0.9]
  --adam_beta2=<adam_beta2>                      [default: 0.999]
  --adadelta_rho=<adadelta_rho>                    [default: 0.95]
  --rmsprop_decay=<rmsprop_decay>                   [default: 0.9]
  --rmsprop_momentum=<rmsprop_momentum>                [default: 0.0]

  --max_epoch=<max_epoch>

  --steps_per_report=<steps_per_report>
  --epochs_per_report=<epochs_per_report>
  --save_checkpoint_networks=<save_checkpoint_networks>
  --save_best_networks=<save_best_networks>
  [-v | -vv]                        Verbosity

Help:
  For help using this tool, please open an issue on the Github repository:
  https://github.com/rdegges/skele-cli
"""


from inspect import getmembers, isclass
from schema import Schema, And, Or, Use

from docopt import docopt
from subprocess import call

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import ast

def main():
    """Main CLI entrypoint."""
    args = docopt(__doc__, options_first=True)
    from IPython import embed
    s = Schema({
        '--adadelta_rho'             : Use(float),
        '--adam_beta1'               : Use(float),
        '--adam_beta2'               : Use(float),
        '--cost_l1_scale'            : Use(float),
        '--cost_l2_scale'            : Use(float),
        '--drop_chance'              : Use(float),
        '--drop_outlier_above'       : Use(float),
        '--validation_fraction'      : Use(float),
        '--rmsprop_decay'            : Use(float),
        '--rmsprop_momentum'         : Use(float),
        '--test_fraction'            : Use(float),
        '--dataset_path'             : str,
        '--dtype'                    : str,
        '--early_stop_after'         : Use(int),
        '--minibatches'              : Use(int),
        '--early_stop_measure'       : 'loss',
        '--epochs_per_report'        : Or(None, Use(int)),
        '--steps_per_report'         : Or(None, Use(int)),
        '--max_epoch'                : Or(None, Use(int)),
        '--save_best_networks'       : Or(None, Use(bool)),
        '--save_checkpoint_networks' : Or(None, Use(bool)),
        '--hidden_activation'        : Use(ast.literal_eval),
        '--hidden_neurons'           : Use(ast.literal_eval),
        '--lbfgs_maxfun'             : Use(int),
        '--lbfgs_maxiter'            : Use(int),
        '--lbfgs_maxls'              : Use(int),
        '--learning_rate'            : Use(float),
        '--optimizer'                : str,
        '--standardization'          : str,
        '--goodness'                 : str,
        '--output_activation'        : str,
        '--train_dims'               : Use(ast.literal_eval)
       },
    ignore_extra_keys=True)

    if args['-v'] >= 2:
        print ('received:')
        print ('global arguments:')
        print (args)

    settings = {}
    for key, val in s.validate(args).items():
        settings[key[2:]] = val
    if args['<command>'] == 'train':
        import train_NDNN
        NNDB_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../NNDB'))
        sys.path.append(NNDB_path)
        from model import Network, TrainScript
        import tempfile
        import shutil
        import json
        old_dir = os.curdir
        with tempfile.TemporaryDirectory() as tmpdirname:
            print('created temporary directory', tmpdirname)
            TrainScript.from_file('./train_NDNN.py')
            shutil.copy('./train_NDNN.py', os.path.join(tmpdirname, 'train_NDNN.py'))
            settings['dataset_path'] = os.path.abspath(settings['dataset_path'])
            with open(os.path.join(tmpdirname, 'settings.json'), 'w') as file_:
                json.dump(settings, file_)
            os.chdir(tmpdirname)
            train_NDNN.train(settings)
            embed()
            Network.from_folder(tmpdirname)
            os.chdir(old_dir)
    if args['<command>'] == 'dict':
        import json
        print(json.dumps(settings))
    elif args['<command>'] in ['help', None]:
        print('<command>')
        exit(call([sys.executable, sys.argv[0], '--help']))
    else:
        exit("{0!s} is not a {1!s} command. See '{1!s} help'.".format(args['<command>'], sys.argv[0]))

if __name__ == '__main__':
    main()
