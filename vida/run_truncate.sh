#!/bin/bash

VIDA="/Users/chenwei/Desktop/Github/ViDa"

# ##
cd $VIDA/vida/data_processing
python read_machineck.py --inpath ../../data/raw_data/Machinek-data/Machinek-PRF-trunc --rxn Machinek-PRF --num-files 400 --outpath ../../data/post_data/Machinek-PRF-trunc/Machinek-PRF.pkl.gz
python preprocess_data.py --inpath ../../data/post_data/Machinek-PRF-trunc/Machinek-PRF.pkl.gz --outpath ../../data/post_data/Machinek-PRF-trunc/preprocess_Machinek-PRF.npz
python  comp_time.py --inpath ../../data/post_data/Machinek-PRF-trunc/preprocess_Machinek-PRF.npz --outpath ../../data/post_data/Machinek-PRF-trunc/time_Machinek-PRF.npz


# ##
cd $VIDA/vida/adjmat
python convert_adj.py --inpath ../../data/post_data/Machinek-PRF-trunc/preprocess_Machinek-PRF.npz --num-strand 3  --seq-path ../../data/post_data/Machinek-PRF-trunc/Machinek-PRF.pkl.gz --outpath ../../data/post_data/Machinek-PRF-trunc/adjmat_Machinek-PRF.npz


# ##
cd $VIDA/vida/scatter_transform
python adj2scatt.py --inpath ../../data/post_data/Machinek-PRF-trunc/adjmat_Machinek-PRF.npz --outpath ../../data/post_data/Machinek-PRF-trunc/scatt_Machinek-PRF.npz


# ##
cd $VIDA/vida/compute_distances
python comp_dist.py --inpath ../../data/post_data/Machinek-PRF-trunc/preprocess_Machinek-PRF.npz --holdtime ../../data/post_data/Machinek-PRF-trunc/time_Machinek-PRF.npz --adjmat ../../data/post_data/Machinek-PRF-trunc/adjmat_Machinek-PRF.npz --outpath ../../data/post_data/Machinek-PRF-trunc/mpt-ged_Machinek-PRF.npz


# ##
cd $VIDA/vida/models
python dataloader.py --predata ../../data/post_data/Machinek-PRF-trunc/preprocess_Machinek-PRF.npz --scatter ../../data/post_data/Machinek-PRF-trunc/scatt_Machinek-PRF.npz --dist ../../data/post_data/Machinek-PRF-trunc/mpt-ged_Machinek-PRF.npz --fconfig ../../data/post_data/Machinek-PRF-trunc/config_template.json --outpath ../../data/post_data/Machinek-PRF-trunc/dataloader_Machinek-PRF.pkl.gz


# ## TRAIN ###
cd $VIDA/vida/models
python train_vida.py --data ../../data/post_data/Machinek-PRF-trunc/dataloader_Machinek-PRF.pkl.gz --fconfig ../../data/post_data/Machinek-PRF-trunc/config_template.json --outpath ../../data/post_data/Machinek-PRF-trunc  