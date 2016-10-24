#!/usr/bin/python

import os
import cPickle as pkl
import neural_networks
import bayesian_parameter_optimization as bpo
from build_kws_data import get_data
from params import nnet_params, hyperparameter_space
from params import TRIAL_DIRECTORY, MODEL_DIRECTORY, MODEL_NAME
from params import target_glob_str, other_glob_str, file_durations_path
from params import MIN_LENGTH, standard_scaler_path

if __name__ == '__main__':
    print("Executing bayesian hyperparameter optimization")

    # Load data and convert it to float 32
    targets, others = get_data(
        target_glob_str, other_glob_str, file_durations_path, MIN_LENGTH)

    # add mean and std to nnet_params
    ss = pkl.load(open(standard_scaler_path, 'rb'))
    nnet_params['offset'] = ss.mean_
    nnet_params['scale'] = ss.scale_

    # Run parameter optimization FOREVER
    bpo.parameter_search(targets, others,
                         nnet_params,
                         hyperparameter_space,
                         os.path.join(TRIAL_DIRECTORY, MODEL_NAME),
                         MODEL_DIRECTORY,
                         neural_networks.train,
                         MODEL_NAME)
