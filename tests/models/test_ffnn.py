import os

from unittest import TestCase, skip
from IPython import embed

from qlknn.models.ffnn import *

test_files_dir = os.path.abspath(os.path.join(__file__, '../../gen2_test_files'))
efi_network_path = os.path.join(test_files_dir, 'network_1393')
efi_div_efe_network_path = os.path.join(test_files_dir, 'network_1440')


input = pd.DataFrame()
scann = 100
input['Ati'] = np.array(np.linspace(2,13, scann))
input['Ti_Te']  = np.full_like(input['Ati'], 1.)
input['An']  = np.full_like(input['Ati'], 2.)
input['Ate']  = np.full_like(input['Ati'], 5.)
input['qx'] = np.full_like(input['Ati'], 0.660156)
input['smag']  = np.full_like(input['Ati'], 0.399902)
input['x']  = np.full_like(input['Ati'], 0.449951)

class QuaLiKizNDNNTestCase(TestCase):

    def setUp(self):
        json_file = os.path.join(efi_network_path, 'nn.json')
        with open(json_file) as file_:
            dict_ = json.load(file_)
        self.nn = QuaLiKizNDNN(dict_, layer_mode='classic')

        super(QuaLiKizNDNNTestCase, self).setUp()


    def tearDown(self):
        super(QuaLiKizNDNNTestCase, self).tearDown()

class TestQuaLiKizNDNN(QuaLiKizNDNNTestCase):
    def test_get_output(self):
        output = self.nn.get_output(input)

class TestQuaLiKizComboNN(TestCase):
    def setUp(self):
        json_file = os.path.join(efi_network_path, 'nn.json')
        with open(json_file) as file_:
            dict_ = json.load(file_)
        self.efi_nn = QuaLiKizNDNN(dict_, layer_mode='classic')

        json_file = os.path.join(efi_div_efe_network_path, 'nn.json')
        with open(json_file) as file_:
            dict_ = json.load(file_)
        self.efi_div_efe_nn = QuaLiKizNDNN(dict_, layer_mode='classic')

        super().setUp()

    def test_create(self):
        QuaLiKizComboNN(['efeITG_GB'],
                        [self.efi_nn, self.efi_div_efe_nn],
                        lambda nn0, nn1: nn0 * nn1)

    def test_get_output(self):
        net = QuaLiKizComboNN(['efeITG_GB'],
                              [self.efi_nn, self.efi_div_efe_nn],
                              lambda nn0, nn1: nn0 * nn1)
        out_combo = net.get_output(input,
                                   clip_low=False,
                                   clip_high=False)
        out_sep = pd.DataFrame(self.efi_nn.get_output(input,
                                                      clip_low=False,
                                                      clip_high=False).values *
                               self.efi_div_efe_nn.get_output(input,
                                                              clip_low=False,
                                                              clip_high=False).values,
                               columns=['efeITG_GB'])
        assert out_combo.equals(out_sep)

    def test_multi_output(self):
        net = QuaLiKizComboNN(['efiITG_GB', 'efeITG_GB_div_efiITG_GB'],
                              [self.efi_nn, self.efi_div_efe_nn],
                              lambda *args: np.hstack(args))
        efi_nn_out = self.efi_nn.get_output(input,
                                            clip_low=False,
                                            clip_high=False)
        efi_div_efe_nn_out = self.efi_div_efe_nn.get_output(input,
                                                            clip_low=False,
                                                            clip_high=False)
        out_sep = efi_nn_out.join(efi_div_efe_nn_out)

        out_combo = net.get_output(input,
                                   clip_low=False,
                                   clip_high=False)
        assert out_combo.equals(out_sep)
