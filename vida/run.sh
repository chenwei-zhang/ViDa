#!/bin/bash

VIDA="/Users/chenwei/Desktop/Github/ViDa"
DATA='Machinek-Mismatch2-trunc'
NAME='Machinek-Mismatch2'


##
cd $VIDA/vida/data_processing
python read_machineck.py --inpath ../../data/raw_data/Machinek-data/$DATA --rxn $NAME --num-files 400 --outpath ../../data/post_data/$DATA/$NAME.pkl.gz
python preprocess_data.py --inpath ../../data/post_data/$DATA/$NAME.pkl.gz --outpath ../../data/post_data/$DATA/preprocess_$NAME.npz
python  comp_time.py --inpath ../../data/post_data/$DATA/preprocess_$NAME.npz --outpath ../../data/post_data/$DATA/time_$NAME.npz


# ##
cd $VIDA/vida/adjmat
python convert_adj.py --inpath ../../data/post_data/$DATA/preprocess_$NAME.npz --num-strand 3 --outpath ../../data/post_data/$DATA/adjmat_$NAME.npz


# ##
cd $VIDA/vida/scatter_transform
python adj2scatt.py --inpath ../../data/post_data/$DATA/adjmat_$NAME.npz --outpath ../../data/post_data/$DATA/scatt_$NAME.npz


# ##
cd $VIDA/vida/compute_distances
python comp_dist.py --inpath ../../data/post_data/$DATA/preprocess_$NAME.npz --holdtime ../../data/post_data/$DATA/time_$NAME.npz --adjmat ../../data/post_data/$DATA/adjmat_$NAME.npz --outpath ../../data/post_data/$DATA/mpt-ged_$NAME.npz


# ##
cd $VIDA/vida/models
python dataloader.py --predata ../../data/post_data/$DATA/preprocess_$NAME.npz --scatter ../../data/post_data/$DATA/scatt_$NAME.npz --dist ../../data/post_data/$DATA/mpt-ged_$NAME.npz --fconfig ../../data/post_data/$DATA/config_template.json --outpath ../../data/post_data/$DATA/dataloader_$NAME.pkl.gz


# ## TRAIN ###
cd $VIDA/vida/models
python train_vida.py --data ../../data/post_data/$DATA/dataloader_$NAME.pkl.gz --fconfig ../../data/post_data/$DATA/config_template.json --outpath ../../data/post_data/$DATA