#!/bin/bash

NAME='gao_p4t4'
NUMFILE=5000

##
python vida/data_processing/read_gao.py --rxn $NAME --num_traj $NUMFILE
python vida/data_processing/preprocess_data.py --rxn $NAME
python vida/data_processing/comp_time.py --rxn $NAME 

##
python vida/adjmat/convert_adj.py --rxn $NAME

#
python vida/compute_distances/comp_dist.py --rxn $NAME 


##
python vida/scatter_transform/adj2scatt.py --rxn $NAME 


##
python vida/models/dataloader.py --rxn $NAME --batch_size 256


# ## TRAIN ###
python vida/models/tune_vida.py --rxn $NAME --trials 10
python vida/models/train_vida.py --rxn $NAME 




# use tensorboard to monitor the training process
# tensorboard --logdir model_config