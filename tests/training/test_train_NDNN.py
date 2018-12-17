from IPython import embed

from qlknn.training.train_NDNN import *
from tests.base import *

class TrainNDNNTestCase(TestCase):
    def setUp(self):
        self.settings = default_train_settings

        #self.train_nn = TrainNN(settings=settings,
        #                        train_dims=['efiITG_GB'],
        #                        uid='test')
        self.test_dir = tempfile.mkdtemp(prefix='test_')
        with open(os.path.join(self.test_dir, 'settings.json'), 'w') as file_:
            settings = json.dump(self.settings, file_)
        #train(settings, warm_start_nn=warm_start_nn)
        os.old_dir = os.curdir
        os.chdir(self.test_dir)

        super(TrainNDNNTestCase, self).setUp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        os.chdir(os.old_dir)
        super(TrainNDNNTestCase, self).setUp()

class TestTrainNN(TrainNDNNTestCase):

    def test_launch_train_NDNN(self):
        print(self.settings.keys())
        train(self.settings)
