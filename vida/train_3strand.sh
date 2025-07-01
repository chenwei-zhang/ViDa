#!/bin/bash

NAME='central_toehold8'
NUMFILE=1000

##
python vida/data_processing/read_machineck.py --rxn $NAME --num_traj $NUMFILE
python vida/data_processing/preprocess_data.py --rxn $NAME 
python vida/data_processing/comp_time.py --rxn $NAME 

##
python vida/adjmat/convert_adj.py --rxn $NAME --num_strand 3 

# ##
python vida/compute_distances/comp_dist.py --rxn $NAME 


##
python vida/scatter_transform/adj2scatt.py --rxn $NAME 


# ##
cp vida/models/config_template.json data/post_data/$NAME/config_template.json
python vida/models/dataloader.py --rxn $NAME 


# # ## TRAIN ###
python vida/models/train_vida.py --rxn $NAME 



# use tensorboard to monitor the training process
# tensorboard --logdir model_config