~/Desktop/kaldi/src/online2bin/online2-wav-nnet2-latgen-faster --do-endpointing=false \
    --online=false \
    --config=nnet_a_gpu_online/conf/online_nnet2_decoding.conf \
    --max-active=7000 --beam=15.0 --lattice-beam=6.0 \
    --acoustic-scale=0.1 --word-symbol-table=graph/words.txt \
   nnet_a_gpu_online/final.mdl graph/HCLG.fst \
   "ark:echo utterance-id1 utterance-id1|" \
   "scp:echo utterance-id1 audio/clinton1_8k.wav|" \
   "ark:|~/Desktop/kaldi/src/latbin/lattice-best-path --acoustic-scale=0.1 ark:- ark,t:- | ~/Desktop/kaldi/egs/fisher_english/s5/utils/int2sym.pl -f 2- graph/words.txt > transcripts/clinton1.txt"


   # online2-wav-nnet2-latgen-faster [options] 
   # <nnet2-in> 
   # <fst-in> 
   # <spk2utt-rspecifier> 
   # <wav-rspecifier> 
   # <lattice-wspecifier>