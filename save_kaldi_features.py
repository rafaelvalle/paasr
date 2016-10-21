import os
import glob2 as glob
from kaldi_nnet_tools import save_kaldi_features
glob_str = "/Users/rafaelvalle/Desktop/speech_data/target_audio/**/*.mfcc"
filepaths = [x for x in glob.glob(os.path.join(glob_str))]
map(save_kaldi_features, filepaths)
