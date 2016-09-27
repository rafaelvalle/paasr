#!/bin/bash
echo "computing $0 with $1"
~/Desktop/kaldi/src/online2bin/online2-wav-nnet2-latgen-faster --do-endpointing=false \
    --online=false \
    --config=conf/online_nnet2_decoding.conf \
    --max-active=7000 --beam=15.0 --lattice-beam=6.0 \
    --acoustic-scale=0.1 --word-symbol-table=graph/words.txt \
    models/fisher_final.mdl graph/HCLG.fst \
    "ark:echo utterance-id1 utterance-id1|" \
    "scp:echo utterance-id1 audio/$1.wav|" \
    "ark:|~/Desktop/kaldi/src/latbin/lattice-best-path --acoustic-scale=0.1 ark:- ark,t:- | ~/Desktop/kaldi/egs/fisher_english/s5/utils/int2sym.pl -f 2- graph/words.txt > transcripts/$1.txt"


   # online2-wav-nnet2-latgen-faster [options] 
   # <nnet2-in> 
   # <fst-in> 
   # <spk2utt-rspecifier> 
   # <wav-rspecifier> 
   # <lattice-wspecifier>
