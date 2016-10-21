#!/bin/bash
filepath=$1
filename=$(basename "$1")
extension="${filename##*.}"
filename="${filename%.*}"
curpath=${PWD}

echo "executing $0 with $1"
~/Desktop/kaldi/src/online2bin/online2-wav-dump-features \
    --ivector-extraction-config=/Users/rafaelvalle/Desktop/pasr/conf/ivector_extractor.conf \
    --config=/Users/rafaelvalle/Desktop/pasr/conf/online_nnet2_dump.conf \
    --verbose=1 \
    "ark:echo utterance-id1 utterance-id1|" \
    "scp:echo utterance-id1 $curpath/$filepath|" \
    "ark,t:$curpath/features/$filename.ark"
echo "done with $0 and $1"    
# online2-wav-dump-features [options] <spk2utt-rspecifier> <wav-rspecifier> <feature-wspecifier>
