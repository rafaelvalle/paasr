#!/bin/bash
~/Desktop/kaldi/src/nnet2bin/nnet-am-compute --apply-log=false \
                /Users/rafaelvalle/Desktop/ipasr/models/fisher_final_txt.mdl \
                "ark,t:/Users/rafaelvalle/Desktop/ipasr/features/clinton1_8k.ark" \
                "ark,t:/Users/rafaelvalle/Desktop/ipasr/log_likelihoods/kaldi_clinton1_8k_ll.ark"