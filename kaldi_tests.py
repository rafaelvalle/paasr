import numpy as np
import kaldi_nnet_tools as knt
# path where model info nnet-am-info and copy nnet-am-copy are saved
am_copy_path = 'models/fisher_final.mdl.nnet.txt'
am_info_path = 'models/fisher_final.mdl.info.txt'

# convert kaldi model to python
net = knt.parseNNET(am_copy_path, am_info_path)

# read context and iVector(constant component) dimension
left_context = abs(net[2]['<Context>'][0])
right_context = abs(net[2]['<Context>'][-1])
const_component_dim = net[2]['<ConstComponentDim>']

"""
# TODO : extract features from python
wav_dump_features_path = (
    "/Users/rafaelvalle//Desktop/kaldi/src/online2bin/online2-wav-dump-features"
options = ("--config=/Users/rafaelvalle//Desktop/kaldi_online_fisher/"+
    "nnet_a_gpu_online/conf/online_nnet2_dump.conf --verbose=1"
spk2utt_rspecifier = "ark:echo utterance-id1 utterance-id1"
wav_rspecifier = "scp:echo utterance-id1 audio/clinton1_8k.wav"
feature_wspecifier = "ark,t:features/clinton1_8k.ark"

data = extract_features(wav_dump_features_path, options, spk2utt_rspecifier,
    wav_rspecifier, feature_wspecifier)
"""

# read kaldi features and transform them into numpy array
feature_path = "features/clinton1_8k.ark"
data = knt.read_kaldi_features(feature_path)

# create generator with spliced data and iVectors
data_gen = knt.splice(data, left_context, right_context, const_component_dim)

# compute ouputs using neural network layers. first two items in list are for
# description
output = [knt.forward(i, net[3:], verbose=False) for i in data_gen]
output = np.clip(output, 1.0e-20, np.inf)
output = np.log(output)
knt.save_kaldi_loglikelihoods(output,
                              'log_likelihoods/clinton1_8k_ll_prior.ark')
