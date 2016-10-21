#!/usr/bin/python

""" Search for good hyperparameters for classifiction using the manually
perturbed (missing data) ADULT and VOTES datasets.
"""
import os
import pdb
import numpy as np
import neural_networks
import bayesian_parameter_optimization as bpo
from build_kws_data import get_data
from params import nnet_params, hyperparameter_space, feats_train_folder, MODEL_NAME
from params import TRIAL_DIRECTORY, MODEL_DIRECTORY
from params import target_glob_str, other_glob_str, file_durations_path, threshold


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

if __name__ == '__main__':
    print("Executing bayesian hyperparameter optimization")

    # Load data and convert it to float 32
    targets, others = get_data(target_glob_str, other_glob_str, file_durations_path, threshold)
    
    # Run parameter optimization FOREVER
    bpo.parameter_search(targets, others,
                         nnet_params,
                         hyperparameter_space,
                         os.path.join(TRIAL_DIRECTORY, MODEL_NAME),
                         MODEL_DIRECTORY,
                         neural_networks.train,
                         MODEL_NAME)
