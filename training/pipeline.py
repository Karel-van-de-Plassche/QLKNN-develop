import sciluigi as sl
import luigi
import train_launch
import train_NDNN
import os
import json
from IPython import embed

class TrainNNWorkflow(sl.WorkflowTask):
    def workflow(self):
        train_nn = self.new_task('train_nn', TrainNN, path='test2')
        return train_nn


class TrainNN(sl.SlurmTask):
    settings = luigi.DictParameter()
    slurminfo = sl.SlurmInfo(sl.RUNMODE_LOCAL, 'tester', 'karelQueue', 1, '20:00:00', 'testjob', 2)
    path = sl.Parameter()

    def out_nn(self):
        return sl.TargetInfo(self, os.path.join('nn.json'))

    def run(self):
        #os.chdir(os.path.dirname(self.in_settings().path))
        cli_settings = []
        for key, val in self.settings.items():
            if isinstance(val, tuple):
                if isinstance(val[0], str):
                    val = '","'.join(map(str, val))
                    val += '"'
                    val = '"' + val
                else:
                    val = ','.join(map(str, val))
            if val is not None:
                cli_settings.append('--' + key + '=' + json.dumps(val))
                #cli_settings['--' + key] = val
        #opt_string = json.dumps(cli_settings)
        #opt_string = opt_string.replace("[", "")
        #opt_string = opt_string.replace("]", "")


        embed()
        self.ex(['./train_NDNN_cli.py', '-vv'] + cli_settings + ['train'])

if __name__ == '__main__':
    sl.run(main_task_cls=TrainNNWorkflow)
