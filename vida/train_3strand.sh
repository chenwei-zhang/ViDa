#!/bin/bash

VIDA="/Users/chenwei/Desktop/Github/ViDa"
# NAME='central_toehold8'
NAME='perfect_toehold7_dangle_CC2GG'
NUMFILE=400

# ##
# cd $VIDA/vida/data_processing
# python read_machineck.py --inpath ../../data/raw_data/machinektest --rxn $NAME --num_traj $NUMFILE --outpath ../../data/post_data/$NAME/$NAME.pkl.gz
# python preprocess_data.py --inpath ../../data/post_data/$NAME/$NAME.pkl.gz --outpath ../../data/post_data/$NAME/preprocess_$NAME.npz
# python comp_time.py --inpath ../../data/post_data/$NAME/preprocess_$NAME.npz --outpath ../../data/post_data/$NAME/time_$NAME.npz

# ##
# cd $VIDA/vida/adjmat
# python convert_adj.py --inpath ../../data/post_data/$NAME/preprocess_$NAME.npz --num_strand 3 --outpath ../../data/post_data/$NAME/adjmat_$NAME.npz

# # ##
# cd $VIDA/vida/compute_distances
# python comp_dist.py --inpath ../../data/post_data/$NAME/preprocess_$NAME.npz --holdtime ../../data/post_data/$NAME/time_$NAME.npz --adjmat ../../data/post_data/$NAME/adjmat_$NAME.npz --outpath ../../data/post_data/$NAME/mpt-ged_$NAME.npz

# ##
# cd $VIDA/vida/scatter_transform
# python adj2scatt.py --inpath ../../data/post_data/$NAME/adjmat_$NAME.npz --outpath ../../data/post_data/$NAME/scatt_$NAME.npz


# # ##
# cd $VIDA/vida/models
# cp $VIDA/data/config_template.json ../../data/post_data/$NAME/config_template.json
# python dataloader.py --predata ../../data/post_data/$NAME/preprocess_$NAME.npz --scatter ../../data/post_data/$NAME/scatt_$NAME.npz --dist ../../data/post_data/$NAME/mpt-ged_$NAME.npz --fconfig ../../data/post_data/$NAME/config_template.json --outpath ../../data/post_data/$NAME/dataloader_$NAME.pkl.gz


# # ## TRAIN ###
cd $VIDA/vida/models
python train_vida.py --data ../../data/post_data/$NAME/dataloader_$NAME.pkl.gz --fconfig ../../data/post_data/$NAME/config_template.json --outpath ../../data/post_data/$NAME




# use tensorboard to monitor the training process
# tensorboard --logdir model_config