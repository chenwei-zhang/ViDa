#!/bin/bash

VIDA="/Users/chenwei/Desktop/Github/ViDa"
DATA='Machinek-PRF-trunc'

### PLOT ###
TNAME='24-0302-1844'
CKPT='checkpoint_epoch_29'
# CKPT='model'


echo "Embedding"
cd $VIDA/vida/models
if [ -f "../../data/post_data/$DATA/model_config/$TNAME/embed_"$CKPT"_Machinek-PRF.npz" ]; then
    echo "Embedding already exists, skip embedding"
else
    python embed_vida.py --data ../../data/post_data/$DATA/dataloader_Machinek-PRF.pkl.gz --model ../../data/post_data/$DATA/model_config/$TNAME/$CKPT.pt --fconfig ../../data/post_data/$DATA/model_config/$TNAME/config.json --outpath ../../data/post_data/$DATA/model_config/$TNAME/embed_"$CKPT"_Machinek-PRF.npz
fi

echo "Plotting"
cd $VIDA/vida/plot
python interact_plot.py --predata ../../data/post_data/$DATA/preprocess_Machinek-PRF.npz --timedata ../../data/post_data/$DATA/time_Machinek-PRF.npz --seqdata ../../data/post_data/$DATA/adjmat_Machinek-PRF.npz --embeddata ../../data/post_data/$DATA/model_config/$TNAME/embed_"$CKPT"_Machinek-PRF.npz --outpath ../../data/post_data/$DATA/model_config/$TNAME/plot_"$CKPT"_Machinek-PRF
