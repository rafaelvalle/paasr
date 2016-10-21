#!/bin/bash
echo "computing $0 with $1"

~/Desktop/kaldi/src/bin/latgen-faster-mapped --acoustic-scale=0.1 \
    --allow-partial=true \
    --word-symbol-table=/Users/rafaelvalle/Desktop/pasr/graph/words.txt \
    /Users/rafaelvalle/Desktop/pasr/models/fisher_final.mdl \
    /Users/rafaelvalle/Desktop/pasr/graph/HCLG.fst \
    "ark,t:$1.ark" \
    "ark:|~/Desktop/kaldi/src/latbin/lattice-best-path --acoustic-scale=0.1 ark:- ark,t:- | ~/Desktop/kaldi/egs/fisher_english/s5/utils/int2sym.pl -f 2- graph/words.txt > transcripts/$1.txt"

# latgen-faster-mapped --max-active=$max_active --max-mem=$max_mem --beam=$beam --lattice-beam=$latbeam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
#     $model 
#     $graphdir/HCLG.fst 
#     "ark:$dir/trans_feature.ark" 
#     "ark:|gzip -c > $dir/lat.JOB.gz" 
# Usage: latgen-faster-mapped [options] 
# trans-model-in # from nnet
# (fst-in|fsts-rspecifier) 
# loglikes-rspecifier 
# lattice-wspecifier [ words-wspecifier [alignments-wspecifier] ]
# "ark:|~/Desktop/kaldi/src/latbin/lattice-best-path --acoustic-scale=0.1 ark:- ark,t:- | ~/Desktop/kaldi/egs/fisher_english/s5/utils/int2sym.pl -f 2- graph/words.txt > transcripts/clinton1.txt"
