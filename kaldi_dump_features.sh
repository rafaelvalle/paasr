~/Desktop/kaldi/src/online2bin/online2-wav-dump-features --ivector-extraction-config=nnet_a_gpu_online/conf/ivector_extractor.conf --config=nnet_a_gpu_online/conf/online_nnet2_dump.conf --verbose=1 "ark:echo utterance-id1 utterance-id1|" "scp:echo utterance-id1 audio/clinton1_8k.wav|" "ark,t:features/clinton1_8k.ark"
# online2-wav-dump-features [options] <spk2utt-rspecifier> <wav-rspecifier> <feature-wspecifier>