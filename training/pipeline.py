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
    for var in ['train_dims', 'dataset_path']:
        if var in settings:
            raise Exception(var, 'should be set seperately, not in the settings dict')

class TrainNN(luigi.contrib.postgres.CopyToTable):
    settings = luigi.DictParameter()
    train_dims = luigi.ListParameter()
    dataset_path = luigi.Parameter()


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
        check_settings_dict(self.settings)
        settings = dict(self.settings)
        settings['train_dims'] = self.train_dims
        settings['dataset_path'] = self.dataset_path
        self.NNDB_nn = train_job(settings)
        super().run()
        print("train_job done")

    def rows(self):
        yield [self.NNDB_nn.id]


class TrainNNpart(TrainNN):
    pass

class TrainBatch(luigi.WrapperTask):
    base_settings = luigi.DictParameter()
    train_dims = luigi.ListParameter()
    #scan = luigi.DictParameter()
    dataset_path = luigi.Parameter()

    def requires(self):
        settings = dict(self.base_settings)
        check_settings_dict(settings)
        yield TrainNN(settings, self.train_dims, self.dataset_path)
        settings['hidden_neurons'] = [100, 100]
        yield TrainNN(settings, self.train_dims, self.dataset_path)



        #embed()
        #self.ex(['./train_NDNN_cli.py', '-vv'] + cli_settings + ['train'])

if __name__ == '__main__':
    sl.run(main_task_cls=TrainNN)
