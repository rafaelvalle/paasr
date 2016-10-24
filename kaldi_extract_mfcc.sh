#!/bin/bash
filepath=$1
filename=$(basename "$1")
filedir=$(dirname "$1")
extension="${filename##*.}"
filename="${filename%.*}"
curpath=${PWD}

echo "executing $0 with $1"
~/Desktop/kaldi/src/featbin/compute-mfcc-feats \
    --config=/Users/rafaelvalle/Desktop/paasr/conf/mfcc.conf \
    --verbose=1 \
    "scp:echo $filename $filepath|" \
    "ark,t:$filedir/$filename.mfcc"
echo "done with $0 and $1"    
# compute-mfcc-feats  $vtln_opts --verbose=2 --config=$mfcc_config \
#      scp,p:$logdir/wav_${name}.JOB.scp
