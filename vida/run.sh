#!/bin/bash

# ##
# cd /Users/chenwei/Desktop/Github/ViDa/vida/data_processing

# python read_machineck.py --inpath ../../data/raw_data/Machinek-data --rxn Machinek-PRF --outpath ../../data/post_data/Machinek-PRF/Machinek-PRF.pkl.gz

# python preprocess_data.py --inpath ../../data/post_data/Machinek-PRF/Machinek-PRF.pkl.gz --outpath ../../data/post_data/Machinek-PRF/preprocess_Machinek-PRF.npz

# python  comp_time.py --inpath ../../data/post_data/Machinek-PRF/preprocess_Machinek-PRF.npz --outpath ../../data/post_data/Machinek-PRF/time_Machinek-PRF.npz


# # ##
# cd /Users/chenwei/Desktop/Github/ViDa/vida/adjmat

# python convert_adj.py --inpath ../../data/post_data/Machinek-PRF/preprocess_Machinek-PRF.npz --num-strand 3  --seq-path ../../data/post_data/Machinek-PRF/Machinek-PRF.pkl.gz --outpath ../../data/post_data/Machinek-PRF/adjmat_Machinek-PRF.npz


# # ##
# cd /Users/chenwei/Desktop/Github/ViDa/vida/scatter_transform

# python adj2scatt.py --inpath ../../data/post_data/Machinek-PRF/adjmat_Machinek-PRF.npz --outpath ../../data/post_data/Machinek-PRF/scatt_Machinek-PRF.npz


# # ##
# cd /Users/chenwei/Desktop/Github/ViDa/vida/compute_distances

# python comp_dist.py --inpath ../../data/post_data/Machinek-PRF/preprocess_Machinek-PRF.npz --holdtime ../../data/post_data/Machinek-PRF/time_Machinek-PRF.npz --adjmat ../../data/post_data/Machinek-PRF/adjmat_Machinek-PRF.npz --outpath ../../data/post_data/Machinek-PRF/mpt-ged_Machinek-PRF.npz



### TRAIN ###
# cp /Users/chenwei/Desktop/Github/ViDa/data/config_template.json /Users/chenwei/Desktop/Github/ViDa/data/post_data/Machinek-PRF/config_template.json

# cd /Users/chenwei/Desktop/Github/ViDa/vida/models

# python dataloader.py --predata ../../data/post_data/Machinek-PRF/preprocess_Machinek-PRF.npz --scatter ../../data/post_data/Machinek-PRF/scatt_Machinek-PRF.npz --dist ../../data/post_data/Machinek-PRF/mpt-ged_Machinek-PRF.npz --fconfig ../../data/post_data/Machinek-PRF/config_template.json --outpath ../../data/post_data/Machinek-PRF/dataloader_Machinek-PRF.pkl.gz

# python train_vida.py --data ../../data/post_data/Machinek-PRF/dataloader_Machinek-PRF.pkl.gz --fconfig ../../data/post_data/Machinek-PRF/config_template.json --outpath ../../data/post_data/Machinek-PRF  


## PLOT ###
TNAME='24-0211-1439'
CKPT='checkpoint_epoch_79'

cd /Users/chenwei/Desktop/Github/ViDa/vida/models
python embed_vida.py --data ../../data/post_data/Machinek-PRF/dataloader_Machinek-PRF.pkl.gz --model ../../data/post_data/Machinek-PRF/model_config/$TNAME/$CKPT.pt --fconfig ../../data/post_data/Machinek-PRF/model_config/$TNAME/config.json --outpath ../../data/post_data/Machinek-PRF/model_config/$TNAME/embed_"$CKPT"_Machinek-PRF.npz

cd /Users/chenwei/Desktop/Github/ViDa/vida/plot
python interact_plot.py --predata ../../data/post_data/Machinek-PRF/preprocess_Machinek-PRF.npz --timedata ../../data/post_data/Machinek-PRF/time_Machinek-PRF.npz --embeddata ../../data/post_data/Machinek-PRF/model_config/$TNAME/embed_"$CKPT"_Machinek-PRF.npz --outpath ../../data/post_data/Machinek-PRF/model_config/$TNAME/plot_"$CKPT"_Machinek-PRF.html
