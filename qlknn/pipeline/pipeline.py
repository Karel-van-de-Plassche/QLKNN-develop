import os
import shutil
import json
import signal
import traceback
import sys
import time
import tempfile
import socket
import re
import subprocess
from subprocess import Popen
from itertools import product

import luigi
import luigi.contrib.postgres
from IPython import embed

from qlknn.NNDB.model import TrainScript, PureNetworkParams, db
import qlknn.training.train_NDNN as train_NDNN
training_path = os.path.dirname(train_NDNN.__file__)

#class TrainNNWorkflow():
#    def workflow(self):
#        train_nn = self.new_task('train_nn', TrainNN, path='test2')
#        return train_nn
if sys.version_info.major < 3:  # Python 2?
    # Using exec avoids a SyntaxError in Python 3.
    exec("""def reraise(exc_type, exc_value, exc_traceback=None):
                raise exc_type, exc_value, exc_traceback""")
else:
    def reraise(exc_type, exc_value, exc_traceback=None):
        if exc_value is None:
            exc_value = exc_type()
        if exc_value.__traceback__ is not exc_traceback:
            raise exc_value.with_traceback(exc_traceback)
        raise exc_value

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
    master_pid = os.getpid()
    sleep_time = 10
    interact_with_nndb = True

    if socket.gethostname().startswith('r0'):
        machine_type = 'marconi'
    elif socket.gethostname().startswith('login'):
        machine_type = 'lisa'
    else:
        machine_type = 'general'


    database = 'nndb'
    host = 'gkdb.org'
    user = 'someone'
    password = 'something'
    table = 'task'
    if interact_with_nndb:
        with open(os.path.join(os.path.expanduser('~'), '.pgpass')) as file_:
            line = file_.read()
            split = line.split(':')
        user=split[-2].strip()
        password=split[-1].strip()
    columns = [('network_id', 'INT')]

    def run_async_io_cmd(self, cmd):
        print(' '.join(cmd))
        proc = Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for stdout_line in iter(proc.stdout.readline, ""):
            yield stdout_line
        proc.stdout.close()
        return_code = proc.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)

    @staticmethod
    def parse_batch_file(prefix, filepath):
        cmd_list = []
        for line in open(filepath, 'r'):
            if line.startswith(prefix):
                cmd = line.lstrip(prefix).strip()
                cmd_list.append(cmd)
        return cmd_list

    def launch_train_NDNN(self):
        if (self.machine_type == 'marconi') or (self.machine_type == 'lisa'):
            print('Starting batch system job')
            pipeline_path = os.path.dirname(os.path.abspath( __file__ ))
            if self.machine_type == 'marconi':
                batch_path = os.path.join(pipeline_path, 'train_NDNN_marconi.sh')
                cmd = ['srun'] + self.parse_batch_file('#SBATCH', batch_path) + [batch_path]
            elif self.machine_type == 'lisa':
                cmd = ['qsub', '-Ix', '-o', 'stdout', '-e', 'stderr', os.path.join(pipeline_path, 'train_NDNN_lisa.sh')]
            try:
                for line in self.run_async_io_cmd(cmd):
                    print(line)
                    if self.machine_type == 'marconi':
                        match = re.match('srun: job (\d*) queued and waiting for resources', line)
                        if match is not None:
                            self.job_id = match.groups()[0] + '.marconi'
                            self.set_status_message_wrapper('Submitted job {!s}, waiting for resources'.format(self.job_id))
                        match = re.match('srun: job (\d*) has been allocated resources', line)
                        if match is not None:
                            self.job_id = match.groups()[0] + '.marconi'
                            self.set_status_message_wrapper('Job {!s} started'.format(self.job_id))

                    elif self.machine_type == 'lisa':
                        match = re.match('qsub: waiting for job (\d*.[\w|.]*)', line)
                        if match is not None:
                            self.job_id = match.groups()[0]
                            self.set_status_message_wrapper('Submitted job {!s}, waiting for resources'.format(self.job_id))
                        match = re.match('qsub: job (\d*.[\w|.]*)', line)
                        if match is not None:
                            self.job_id = match.groups()[0]
                            self.set_status_message_wrapper('Job {!s} started'.format(self.job_id))

            except subprocess.CalledProcessError as err:
                print('Error in job process! waiting for stdout/stderr')
                import time
                exc_type = type(err)
                exc_traceback = sys.exc_info()[2]
                timeout = 60
                sleep_time = 1
                exc_value = 'STDOUT:\n'
                for ii in range(timeout):
                    try:
                        with open('stdout') as file_:
                            exc_value += file_.read()
                    except IOError:
                        time.sleep(sleep_time)
                    else:
                        break
                exc_value += 'STDERR:\n'
                for ii in range(timeout):
                    try:
                        with open('stderr') as file_:
                            exc_value += file_.read()
                    except IOError:
                        time.sleep(sleep_time)
                    else:
                        break
                new_exc = RuntimeError(exc_value)
                reraise(new_exc.__class__, new_exc, exc_traceback)
            #cmd = ' '.join(['python train_NDNN.py'])
            #subprocess.check_call(cmd, shell=True, stdout=None, stderr=None)
        else:
            self.job_id = 'local'
            print('Starting local job')
            train_NDNN.train_NDNN_from_folder()

    def run(self):
        self.set_status_message_wrapper('Starting job')
        os.chdir(os.path.dirname(__file__))
        check_settings_dict(self.settings)
        settings = dict(self.settings)
        settings['train_dims'] = self.train_dims
        old_dir = os.getcwd()
        if self.machine_type == 'marconi':
            tmproot = os.environ['CINECA_SCRATCH']
        elif self.machine_type == 'lisa':
            tmproot = os.path.join(os.environ['HOME'], 'tmp_nn')
        else:
            tmproot = None
        self.tmpdirname = tmpdirname = tempfile.mkdtemp(prefix='trainNN_', dir=tmproot)
        print('created temporary directory', tmpdirname)
        train_script_path = os.path.join(training_path, 'train_NDNN.py')
        if self.interact_with_nndb:
            db.connect(reuse_if_open=True)
            TrainScript.from_file(train_script_path)
            db.close()
        #shutil.copy(os.path.join(train_script_path), os.path.join(tmpdirname, 'train_NDNN.py'))
        os.symlink(os.path.join(train_script_path), os.path.join(tmpdirname, 'train_NDNN.py'))
        settings['dataset_path'] = os.path.abspath(settings['dataset_path'])
        with open(os.path.join(tmpdirname, 'settings.json'), 'w') as file_:
            json.dump(settings, file_, indent=4, sort_keys=True)
        os.chdir(tmpdirname)
        self.set_status_message_wrapper('Started training on {!s}'.format(socket.gethostname()))
        self.launch_train_NDNN()
        print('Training done!')

        if self.interact_with_nndb:
            if os.path.isfile(os.path.join(tmpdirname, 'nn.json')):
                for ii in range(10):
                    self.set_status_message_wrapper('Trying to submit to NNDB, try: {!s} / 10 on {!s}'.format(ii+1, socket.gethostname()))
                    try:
                        db.connect(reuse_if_open=True)
                        self.NNDB_nn = PureNetworkParams.from_folder(tmpdirname)
                        db.close()
                    except Exception as ee:
                        exception = ee
                        time.sleep(self.sleep_time)
                    else:
                        break
            else:
                exception = Exception('Could not find nn.json! Did training really finish?')
                raise exception
            if not hasattr(self, 'NNDB_nn'):
                raise reraise(type(exception), exception, sys.exc_info()[2])
            else:
                os.chdir(old_dir)
                shutil.rmtree(tmpdirname)
                super(TrainNN, self).run()
                self.set_status_message_wrapper('Done! NNDB id: {!s}'.format(self.NNDB_nn.id))
                print("train_job done")

    def rows(self):
        yield [self.NNDB_nn.id]

    def on_failure(self, exception):
        print('Training failed! Killing master {!s} of worker {!s}'.format(self.master_pid, os.getpid()))
        os.kill(self.master_pid, signal.SIGUSR1)
        os.kill(os.getpid(), signal.SIGUSR1)
        traceback_string = traceback.format_exc()
        with open('traceback.dump', 'w') as file_:
           file_.write(traceback.format_exc())

        message_list = []
        if hasattr(self, 'job_id'):
            message_list.append('JobID: {!s}\n'.format(self.job_id))
        message_list.append('Host: {!s}\nDir: {!s}\nRuntime error:\n{!s}'.format(socket.gethostname(),
                                                                        self.tmpdirname,
                                                                        traceback_string))
        message = ''.join(message_list)
        self.set_status_message_wrapper(message)
        return message

    def on_success(self):
        print('Training success!')
        #print('Killing master {!s} of worker {!s}'.format(self.master_pid, os.getpid()))
        #os.kill(os.getpid(), signal.SIGUSR1)
        #os.kill(self.master_pid, signal.SIGUSR1)

    def set_status_message_wrapper(self, message):
        if self.set_status_message is None:
            print(message)
        else:
            self.set_status_message(message)

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

    with open(os.path.join(training_path, 'default_settings.json')) as file_:
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

class TrainMidsize7DBatch(TrainBatch):
    dim = 7
    gen = 3
    plan = {'cost_l2_scale': [2e-5, 5e-5, 8e-5],
            'hidden_neurons': [[128] * 3],
            'filter': [8],
            'activations': ['tanh'],
            }

    plan['dataset_path'] = []
    for filter in plan.pop('filter'):
        plan['dataset_path'].append('../../unstable_training_gen{!s}_{!s}D_nions0_flat_filter{!s}.h5'.format(gen, dim, filter))

    with open(os.path.join(training_path, 'default_settings.json')) as file_:
        settings = json.load(file_)
        settings.pop('train_dims')

    settings_list = []
    for val in product(*plan.values()):
        par = dict(zip(plan.keys(), val))
        par['hidden_activation'] = [par.pop('activations')] * len(par['hidden_neurons'])
        settings.update(par)
        settings_list.append(settings.copy())

class TrainNarrow9DBatch(TrainBatch):
    dim = 9
    gen = 3
    plan = {'cost_l2_scale': [2e-5, 5e-5, 8e-5],
            'hidden_neurons': [[128] * 3],
            'filter': [8],
            'activations': ['tanh'],
            }

    plan['dataset_path'] = []
    for filter in plan.pop('filter'):
        plan['dataset_path'].append('../unstable_training_gen{!s}_{!s}D_nions0_flat_filter{!s}.h5'.format(gen, dim, filter))

    with open(os.path.join(training_path, 'default_settings.json')) as file_:
        settings = json.load(file_)
        settings.pop('train_dims')

    settings_list = []
    for val in product(*plan.values()):
        par = dict(zip(plan.keys(), val))
        par['hidden_activation'] = [par.pop('activations')] * len(par['hidden_neurons'])
        settings.update(par)
        settings_list.append(settings.copy())

class TrainAll9DNetworks(luigi.WrapperTask):
    submit_date = luigi.DateHourParameter()

    def requires(self):
        for train_dims in gen3_single_target_list:
            yield TrainNarrow9DBatch(self.submit_date, train_dims)

class TrainAll7DNetworks(luigi.WrapperTask):
    submit_date = luigi.DateHourParameter()

    def requires(self):
        for train_dims in gen3_single_target_list:
            yield TrainMidsize7DBatch(self.submit_date, train_dims)

def target_names_generator():
    for mode in ['', 'ITG', 'TEM']:
        type = 'ef'
        for op in ['plus', 'div']:
            name = type + 'i' + mode + '_GB_' + op + '_' + type + 'e' + mode + '_GB'
            yield [name]
        name = type + 'e' + mode + '_GB_' + 'div' + '_' + type + 'i' + mode + '_GB'
        yield [name]
        for species in ['e', 'i']:
            name = type + species + mode + '_GB'
            yield [name]

        name = 'pf' + 'e' + mode + '_GB_' + 'div' + '_' + 'ef' + 'i' + mode + '_GB'
        yield [name]

    name = type + 'eETG_GB'
    yield [name]

gen3_single_target_list = [
    ['efeETG_GB'],
    ['efeITG_GB'],
    ['efiITG_GB'],
    ['efeITG_GB_div_efiITG_GB'],
    ['pfeITG_GB'],
    ['pfeITG_GB_div_efiITG_GB'],
    ['efeTEM_GB'],
    ['efiTEM_GB'],
    ['efiTEM_GB_div_efeTEM_GB'],
    ['pfeTEM_GB'],
    ['pfeTEM_GB_div_efeTEM_GB']
]

gen3_multiD_target_list = [
    ['efeITG_GB', 'efiITG_GB'],
    ['efeTEM_GB', 'efiTEM_GB'],
    ['efeITG_GB', 'efiITG_GB', 'pfeITG_GB'],
    ['efeTEM_GB', 'efiTEM_GB', 'pfeTEM_GB']
]
gen3_target_list = gen3_single_target_list + gen3_multiD_target_list


if __name__ == '__main__':
    luigi.run(main_task_cls=TrainNN)
