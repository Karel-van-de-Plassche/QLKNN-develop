import luigi
import luigi.contrib.postgres
from train_launch import train_job
import train_NDNN
import os
import json
import signal
from IPython import embed
import sys
NNDB_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../NNDB'))
sys.path.append(NNDB_path)
import model
from itertools import product
import time

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
    uid = luigi.Parameter()


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

    def run(self):
        os.chdir(os.path.dirname(__file__))
        check_settings_dict(self.settings)
        settings = dict(self.settings)
        settings['train_dims'] = self.train_dims
        old_dir = os.getcwd()
        tmpdirname = tempfile.mkdtemp(prefix='trainNN_')
        print('created temporary directory', tmpdirname)
        TrainScript.from_file('./train_NDNN.py')
        shutil.copy(os.path.join(os.getcwd(), './train_NDNN.py'), os.path.join(tmpdirname, 'train_NDNN.py'))
        settings['dataset_path'] = os.path.abspath(settings['dataset_path'])
        with open(os.path.join(tmpdirname, 'settings.json'), 'w') as file_:
            json.dump(settings, file_)
        os.chdir(tmpdirname)
        train_NDNN.train(settings)
        print('Training done!')
        for ii in range(10):
            self.set_status_message("Try: {!s} / 10".format(ii))
            try:
                self.NNDB_nn = Network.from_folder(tmpdirname)
            except Exception as ee:
                print(ee)
                time.sleep(5*60)
        os.chdir(old_dir)
        shutil.rmtree(tmpdirname)
        super().run()
        print("train_job done")

    def rows(self):
        yield [self.NNDB_nn.id]

    def on_failure(self, exception):
        print('Training failed! Killing worker')
        os.kill(os.getpid(), signal.SIGUSR1)
        traceback_string = traceback.format_exc()
        return "Runtime error:\n%s" % traceback_string

class TrainBatch(luigi.WrapperTask):
    submit_date = luigi.DateHourParameter()
    train_dims = luigi.ListParameter()
    #scan = luigi.DictParameter()
    settings_list = luigi.ListParameter()

    def requires(self):
        for settings in self.settings_list:
            check_settings_dict(settings)
            yield TrainNN(settings, self.train_dims, self.task_id)

class TrainRepeatingBatch(luigi.WrapperTask):
    submit_date = luigi.DateHourParameter()
    train_dims = luigi.ListParameter()
    #scan = luigi.DictParameter()
    settings = luigi.DictParameter()
    repeat = luigi.IntParameter(significant=False)

    def requires(self):
        check_settings_dict(self.settings)
        for ii in range(self.repeat):
            yield TrainNN(self.settings, self.train_dims, self.task_id + '_' + str(ii))

class TrainDenseBatch(TrainBatch):
    dim = 7
    plan = {'cost_l2_scale': [0.05, 0.1, 0.2],
            'hidden_neurons': [[30] * 3, [64] * 3, [60] * 2],
            'filter': [2, 5],
            'activations': ['tanh', 'relu']
            }

    plan['dataset_path'] = []
    for filter in plan.pop('filter'):
        plan['dataset_path'].append('../filtered_{!s}D_nions0_flat_filter{!s}.h5'.format(dim, filter))

    with open(os.path.join(os.path.dirname(__file__), 'default_settings.json')) as file_:
        settings = json.load(file_)
        settings.pop('train_dims')

    settings_list = []
    for val in product(*plan.values()):
        par = dict(zip(plan.keys(), val))
        if par['activations'] == 'relu':
            par['early_stop_after'] = 15
        par['hidden_activation'] = [par.pop('activations')] * len(par['hidden_neurons'])
        settings.update(par)
        settings_list.append(settings.copy())



        #embed()
        #self.ex(['./train_NDNN_cli.py', '-vv'] + cli_settings + ['train'])

if __name__ == '__main__':
    luigi.run(main_task_cls=TrainNN)
