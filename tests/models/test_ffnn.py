import os

from unittest import TestCase, skip
from IPython import embed

from qlknn.models.ffnn import *

test_files_dir = os.path.abspath(os.path.join(__file__, '../../gen2_test_files'))
efi_network_path = os.path.join(test_files_dir, 'network_1393')
efi_div_efe_network_path = os.path.join(test_files_dir, 'network_1440')

class QuaLiKizNDNNTestCase(TestCase):

    def setUp(self):
        json_file = os.path.join(efi_network_path, 'nn.json')
        with open(json_file) as file_:
            dict_ = json.load(file_)
        self.nn = QuaLiKizNDNN(dict_, layer_mode='classic')

        scann = 100
        input = pd.DataFrame()
        input['Ati'] = np.array(np.linspace(2,13, scann))
        input['Ti_Te']  = np.full_like(input['Ati'], 1.)
        input['An']  = np.full_like(input['Ati'], 2.)
        input['Ate']  = np.full_like(input['Ati'], 5.)
        input['qx'] = np.full_like(input['Ati'], 0.660156)
        input['smag']  = np.full_like(input['Ati'], 0.399902)
        input['x']  = np.full_like(input['Ati'], 0.449951)

        self.input = input
        super(QuaLiKizNDNNTestCase, self).setUp()


    def tearDown(self):
        super(QuaLiKizNDNNTestCase, self).tearDown()

class TestQuaLiKizNDNN(QuaLiKizNDNNTestCase):
    def test_get_output(self):
        output = self.nn.get_output(self.input)
