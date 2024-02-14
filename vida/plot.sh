#!/bin/bash

VIDA="/Users/chenwei/Desktop/Github/ViDa"

### PLOT ###
TNAME='24-0212-2006'
# CKPT='checkpoint_epoch_29'
CKPT='model'

cd $VIDA/vida/models
python embed_vida.py --data ../../data/post_data/Machinek-PRF/dataloader_Machinek-PRF.pkl.gz --model ../../data/post_data/Machinek-PRF/model_config/$TNAME/$CKPT.pt --fconfig ../../data/post_data/Machinek-PRF/model_config/$TNAME/config.json --outpath ../../data/post_data/Machinek-PRF/model_config/$TNAME/embed_"$CKPT"_Machinek-PRF.npz

cd $VIDA/vida/plot
python interact_plot.py --predata ../../data/post_data/Machinek-PRF/preprocess_Machinek-PRF.npz --timedata ../../data/post_data/Machinek-PRF/time_Machinek-PRF.npz --embeddata ../../data/post_data/Machinek-PRF/model_config/$TNAME/embed_"$CKPT"_Machinek-PRF.npz --outpath ../../data/post_data/Machinek-PRF/model_config/$TNAME/plot_"$CKPT"_Machinek-PRF
