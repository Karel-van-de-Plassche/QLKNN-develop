import numpy as np
import pandas as pd
from IPython import embed
from keras.models import load_model
from keras import backend as K

from qlknn.models.ffnn import determine_settings, _prescale, clip_to_bounds

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square( y_true-y_pred )))

class KerasNDNN():
    def __init__(self, model, feature_names, target_names,
                 feature_prescale_factor, feature_prescale_bias,
                 target_prescale_factor, target_prescale_bias,
                 feature_min=None, feature_max=None,
                 target_min=None, target_max=None,
                 target_names_mask=None,
                 ):
        self.model = model
        self._feature_names = feature_names
        self._target_names = target_names
        self._feature_prescale_factor = feature_prescale_factor
        self._feature_prescale_bias = feature_prescale_bias
        self._target_prescale_factor = target_prescale_factor
        self._target_prescale_bias = target_prescale_bias

        if feature_min is None:
            feature_min = pd.Series({var: -np.inf for var in self._feature_names})
        self._feature_min = feature_min
        if feature_max is None:
            feature_max = pd.Series({var: np.inf for var in self._feature_names})
        self._feature_max = feature_max
        if target_min is None:
            target_min = pd.Series({var: -np.inf for var in self._target_names})
        self._target_min = target_min
        if target_max is None:
            target_max = pd.Series({var: np.inf for var in self._target_names})
        self._target_max = target_max
        self._target_names_mask = target_names_mask

    def get_output(self, inp, clip_low=False, clip_high=False, low_bound=None, high_bound=None, safe=True, output_pandas=True, shift_output_by=0):
        """
        This should accept a pandas dataframe, and should return a pandas dataframe
        """
        nn_input, safe, clip_low, clip_high, low_bound, high_bound = \
            determine_settings(self, inp, safe, clip_low, clip_high, low_bound, high_bound)

        nn_input = _prescale(nn_input,
                             self._feature_prescale_factor.values,
                             self._feature_prescale_bias.values)

        # Apply all NN layers an re-scale the outputs
        branched_in = [nn_input.loc[:, self._branch1_names].values,
                                     nn_input.loc[:, self._branch2_names].values]
        nn_out = self.model.predict(branched_in) # Get prediction
        output = (nn_out - np.atleast_2d(self._target_prescale_bias)) / np.atleast_2d(self._target_prescale_factor)
        output -= shift_output_by
        output = clip_to_bounds(output, clip_low, clip_high, low_bound, high_bound)

        if output_pandas:
            output = pd.DataFrame(output, columns=self._target_names)

        if self._target_names_mask is not None:
            output.columns = self._target_names_mask
        return output


class Daniel7DNN(KerasNDNN):
    _branch1_names = ['Ati', 'An', 'q', 'smag', 'x', 'Ti_Te']
    _branch2_names = ['Ate']

    @classmethod
    def from_files(cls, model_file, standardization_file):
        model = load_model(model_file, custom_objects={'rmse': rmse})
        stds = pd.read_csv(standardization_file)
        feature_names = pd.Series(cls._branch1_names + cls._branch2_names)
        target_names = pd.Series(['efeETG_GB'])
        stds.set_index('name', inplace=True)
        # Was normalised to s=1, m=0
        s_t = 1
        m_t = 0
        s_sf = stds.loc[feature_names, 'std']
        s_st = stds.loc[target_names, 'std']
        m_sf = stds.loc[feature_names, 'mean']
        m_st = stds.loc[target_names, 'mean']
        feature_scale_factor = s_t / s_sf
        feature_scale_bias = -m_sf * s_t / s_sf + m_t
        target_scale_factor = s_t / s_st
        target_scale_bias = -m_st * s_t / s_st + m_t
        return cls(model, feature_names, target_names,
                   feature_scale_factor, feature_scale_bias,
                   target_scale_factor, target_scale_bias,
                   )

    def get_output(self, inp, clip_low=False, clip_high=False, low_bound=None, high_bound=None, safe=True, output_pandas=True, shift_output=True):
        if shift_output:
            if not hasattr(self, 'shift'):
                self.shift = self.find_shift()
            shift_output_by = self.shift
        else:
            shift_output_by = 0
        output = super().get_output(inp, clip_low=clip_low, clip_high=clip_high, low_bound=low_bound, high_bound=high_bound, safe=safe, output_pandas=output_pandas, shift_output_by=shift_output_by)
        return output


    def find_shift(self):
        # Define a point where the relu is probably 0
        nn_input = pd.DataFrame({'Ati': 0, 'An': 0, 'q': 3, 'smag': 3, 'x': 0.7, 'Ti_Te': 1, 'Ate': 0}, index=[0])
        branched_in = [nn_input.loc[:, self._branch1_names].values,
                                     nn_input.loc[:, self._branch2_names].values]
        # Get a function to evaluate the network up until the relu layer
        try:
            func = K.function(self.model.input, [self.model.get_layer('TR').output])
        except ValueError:
            raise Exception("'TR' layer not defined, shifting only relevant for new-style NNs")
        relu_out = func(branched_in)
        if relu_out[0][0, 0] != 0:
            raise Exception('Relu is not zero at presumed stable point! Cannot find shift')
        nn_out = self.model.predict(branched_in)
        output = (nn_out - np.atleast_2d(self._target_prescale_bias)) / np.atleast_2d(self._target_prescale_factor)
        shift = output[0][0]
        return shift

if __name__ == '__main__':
    # Test the function
    nn = Daniel7DNN.from_files('2018-12-04_Run0161h-Mk5.h5', 'standardizations_training.csv')
    shift = nn.find_shift()

    scann = 200
    input = pd.DataFrame()
    input['Ate'] = np.array(np.linspace(0,14, scann))
    input['Ti_Te']  = np.full_like(input['Ate'], 1.33)
    input['An']  = np.full_like(input['Ate'], 3.)
    input['Ati']  = np.full_like(input['Ate'], 5.75)
    input['q'] = np.full_like(input['Ate'], 3)
    input['smag']  = np.full_like(input['Ate'], 0.7)
    input['x']  = np.full_like(input['Ate'], 0.45)
    input = input[nn._feature_names]

    fluxes = nn.get_output(input)
    print(fluxes)
    embed()
