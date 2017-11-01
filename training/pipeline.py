import sciluigi as sl
import luigi
from train_launch import train_job
import train_NDNN
import os
import json
from IPython import embed
import sys
NNDB_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../NNDB'))
sys.path.append(NNDB_path)
import model

#class TrainNNWorkflow():
#    def workflow(self):
#        train_nn = self.new_task('train_nn', TrainNN, path='test2')
#        return train_nn

def check_settings_dict(settings):
    for var in ['train_dims']:
        if var in settings:
            raise Exception(var, 'should be set seperately, not in the settings dict')
class DummyTask(luigi.Task):
    pass

class TrainNN(luigi.contrib.postgres.CopyToTable):
    settings = luigi.DictParameter()
    train_dims = luigi.ListParameter()
    batch = luigi.Parameter()


    database = 'nndb'
    host = 'gkdb.org'
    password = 'something'
    table = 'task'
    with open(os.path.join(os.path.expanduser('~'), '.pgpass')) as file_:
        line = file_.read()
        split = line.split(':')
    user=split[-2].strip()
    password=split[-1].strip()
    columns = [('network_id', 'INT')]
    #slurminfo = sl.SlurmInfo(sl.RUNMODE_LOCAL, 'tester', 'karelQueue', 1, '20:00:00', 'testjob', 2)
    #path = sl.Parameter()

    def run(self):
        #os.chdir(os.path.dirname(self.in_settings().path))
        #cli_settings = []
        #for key, val in self.settings.items():
        #    if isinstance(val, tuple):
        #        if isinstance(val[0], str):
        #            val = '","'.join(map(str, val))
        #            val += '",'
        #            val = '"' + val
        #        else:
        #            val = ','.join(map(str, val))
        #    if val is not None:
        #        cli_settings.append('--' + key + '=' + json.dumps(val))
        #        #cli_settings['--' + key] = val
        #opt_string = json.dumps(cli_settings)
        #opt_string = opt_string.replace("[", "")
        #opt_string = opt_string.replace("]", "")
        os.chdir(os.path.dirname(__file__))
        check_settings_dict(self.settings)
        settings = dict(self.settings)
        settings['train_dims'] = self.train_dims
        self.NNDB_nn = train_job(settings)
        super().run()
        print("train_job done")

    def rows(self):
        yield [self.NNDB_nn.id]

class TrainBatch(luigi.WrapperTask):
    submit_date = luigi.DateHourParameter()
    train_dims = luigi.ListParameter()
    #scan = luigi.DictParameter()
    settings_list = luigi.ListParameter()

    def requires(self):
        for settings in self.settings_list:
            check_settings_dict(settings)
            yield TrainNN(settings, self.train_dims, self.task_id)

class TrainDenseBatch(TrainBatch):
    dim = 7
    plan = {'cost_l2_scale': [0.05, 0.1, 0.2],
            'hidden_neurons': [[30] * 3, [64] * 3, [60] * 2],
            'filter': [3, 5],
            'activations': ['tanh', 'relu']
            }

    plan['dataset_path'] = []
    for filter in plan.pop('filter'):
        plan['dataset_path'].append('../filtered_{!s}D_nions0_flat_filter{!s}.h5'.format(dim, filter))

    from itertools import product
    with open('default_settings.json') as file_:
        settings = json.load(file_)
        settings.pop('train_dims')

    settings_list = []
    for val in product(*plan.values()):
        par = dict(zip(plan.keys(), val))
        par['hidden_activation'] = [par.pop('activations')] * len(par['hidden_neurons'])
        settings.update(par)
        settings_list.append(settings.copy())



        #embed()
        #self.ex(['./train_NDNN_cli.py', '-vv'] + cli_settings + ['train'])

if __name__ == '__main__':
    sl.run(main_task_cls=TrainNN)
