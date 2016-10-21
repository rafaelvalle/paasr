import os
import numpy as np
import lasagne

# random seed
rand_num_seed = 1

# KEYWORD
KEYWORD = 'HELP'

# feature configuration
LEFT_CONTEXT = 30
RIGHT_CONTEXT = 10
FRAME_SIZE = .025  # in ms
SAMPLE_LENGTH = (LEFT_CONTEXT + 1 + RIGHT_CONTEXT) * FRAME_SIZE

# data
text_grid_glob_str = '/Users/rafaelvalle/Desktop/speech_data/**/*.TextGrid'
target_glob_str = '/Users/rafaelvalle/Desktop/speech_data/target_audio/help/*.npy'
other_glob_str = '/Users/rafaelvalle/Desktop/speech_data/**/*.npy'
file_durations_path = '/Users/rafaelvalle/Desktop/speech_data/file_duration'

# folder paths
IMAGES_DIRECTORY = "images/"
RESULTS_PATH = 'results/'
TRIAL_DIRECTORY = os.path.join(RESULTS_PATH, 'parameter_trials')
MODEL_DIRECTORY = os.path.join(RESULTS_PATH, 'model')
MODEL_NAME = 'model'
TARGET_AUDIO_DIRECTORY = '/Users/rafaelvalle/Desktop/speech_data/target_audio/help/'


# neural network structure
nnet_params = {'n_folds': 1,
               'n_layers': 4,
               'batch_size': 16,
               'epoch_size': 128,
               'gammas': np.array([0.1, 0.01], dtype=np.float32),
               'decay_rate': 0.95,
               'max_epoch': 50,
               'widths': [None, 512, 512, 2],
               'non_linearities': (None,
                                   lasagne.nonlinearities.rectify,
                                   lasagne.nonlinearities.rectify,
                                   lasagne.nonlinearities.softmax),
               'update_func': lasagne.updates.adadelta,
               'drops': (None, 0.2, 0.5, None)}

# hyperparameter space to be explored using bayesian hyperparameter optimization
hyperparameter_space = {
    'momentum': {'type': 'float', 'min': 0., 'max': 1.},
    'dropout': {'type': 'int', 'min': 0, 'max': 1},
    'learning_rate': {'type': 'float', 'min': .000001, 'max': .01},
    'network': {'type': 'enum', 'options': ['general_network']}
    }
