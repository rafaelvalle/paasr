#!/bin/bash
echo "executing $0 with $1"
~/Desktop/kaldi/src/online2bin/online2-wav-dump-features \
    --ivector-extraction-config=/Users/rafaelvalle/Desktop/ipasr/conf/ivector_extractor.conf \
    --config=/Users/rafaelvalle/Desktop/ipasr/conf/online_nnet2_dump.conf \
    --verbose=1 \
    "ark:echo utterance-id1 utterance-id1|" \
    "scp:echo utterance-id1 /Users/rafaelvalle/Desktop/ipasr/audio/$1.wav|" \
    "ark,t:/Users/rafaelvalle/Desktop/ipasr/features/$1.ark"
echo "done with $0 and $1"    
# online2-wav-dump-features [options] <spk2utt-rspecifier> <wav-rspecifier> <feature-wspecifier>
