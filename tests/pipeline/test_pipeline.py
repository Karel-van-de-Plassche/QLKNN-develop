import os
import shutil
import tempfile

from unittest import TestCase, skip
from IPython import embed

from qlknn.pipeline.pipeline import *
from tests.base import *

class TrainNNTestCase(TestCase):
    def setUp(self):
        self.settings = default_train_settings.copy()
        self.settings.pop('train_dims')

        self.test_dir = tempfile.mkdtemp(prefix='test_')

        self.train_nn = TrainNN(settings=self.settings,
                train_dims=['efiITG_GB'],
                uid = 'test')
        self.train_nn.interact_with_nndb = False
        os.old_dir = os.curdir
        os.chdir(self.test_dir)

        super(TrainNNTestCase, self).setUp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        os.chdir(os.old_dir)
        super(TrainNNTestCase, self).setUp()

class TestDummyTask(TestCase):
    def test_create(self):
        task = DummyTask()

    def test_run(self):
        task = DummyTask()
        task.run()

class TestTrainNN(TrainNNTestCase):

    def test_launch_train_NN(self):
        self.settings['train_dims'] = self.train_nn.train_dims
        with open(os.path.join(self.test_dir, 'settings.json'), 'w') as file_:
            json.dump(self.settings, file_)
        self.train_nn.launch_train_NDNN()

    def test_run(self):
        self.train_nn.sleep_time = 0
        self.train_nn.run()
