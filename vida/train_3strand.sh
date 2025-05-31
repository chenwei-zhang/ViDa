#!/bin/bash

VIDA="/Users/chenwei/Desktop/Github/ViDa"
NAME='central_toehold8'
NUMFILE=1000

##
cd $VIDA/vida/data_processing
python read_machineck.py --inpath $VIDA/data/raw_data/machinektest --rxn $NAME --num_traj $NUMFILE --outpath $VIDA/data/post_data/$NAME/$NAME.pkl.gz
python preprocess_data.py --inpath $VIDA/data/post_data/$NAME/$NAME.pkl.gz --outpath $VIDA/data/post_data/$NAME/preprocess_$NAME.npz
python comp_time.py --inpath $VIDA/data/post_data/$NAME/preprocess_$NAME.npz --outpath $VIDA/data/post_data/$NAME/time_$NAME.npz

##
cd $VIDA/vida/adjmat
python convert_adj.py --inpath $VIDA/data/post_data/$NAME/preprocess_$NAME.npz --num_strand 3 --outpath $VIDA/data/post_data/$NAME/adjmat_$NAME.npz

# ##
cd $VIDA/vida/compute_distances
python comp_dist.py --inpath $VIDA/data/post_data/$NAME/preprocess_$NAME.npz --holdtime $VIDA/data/post_data/$NAME/time_$NAME.npz --adjmat $VIDA/data/post_data/$NAME/adjmat_$NAME.npz --outpath $VIDA/data/post_data/$NAME/mpt-ged_$NAME.npz

##
cd $VIDA/vida/scatter_transform
python adj2scatt.py --inpath $VIDA/data/post_data/$NAME/adjmat_$NAME.npz --outpath $VIDA/data/post_data/$NAME/scatt_$NAME.npz


# ##
cd $VIDA/vida/models
cp config_template.json $VIDA/data/post_data/$NAME/config_template.json
python dataloader.py --predata $VIDA/data/post_data/$NAME/preprocess_$NAME.npz --scatter $VIDA/data/post_data/$NAME/scatt_$NAME.npz --dist $VIDA/data/post_data/$NAME/mpt-ged_$NAME.npz --fconfig $VIDA/data/post_data/$NAME/config_template.json --outpath $VIDA/data/post_data/$NAME/dataloader_$NAME.pkl.gz


# # ## TRAIN ###
cd $VIDA/vida/models
python train_vida.py --data $VIDA/data/post_data/$NAME/dataloader_$NAME.pkl.gz --fconfig $VIDA/data/post_data/$NAME/config_template.json --outpath $VIDA/data/post_data/$NAME




# use tensorboard to monitor the training process
# tensorboard --logdir model_config