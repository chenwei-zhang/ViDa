#!/bin/bash

VIDA="/Users/chenwei/Desktop/Github/ViDa"
# NAME='Machinek-PRF'
NAME='Machinek-Mismatch2'



### PLOT ###
TNAME='24-0701-1546'  # 24-0324-0120
CKPT='checkpoint_epoch_59'
# CKPT='model'


echo "Embedding"
cd $VIDA/vida/models
if [ -f "../../data/post_data/$NAME/model_config/$TNAME/embed_"$CKPT"_"$NAME".npz" ]; then
    echo "Embedding already exists, skip embedding"
else
    python embed_vida.py --data ../../data/post_data/$NAME/dataloader_"$NAME".pkl.gz --model ../../data/post_data/$NAME/model_config/$TNAME/$CKPT.pt --fconfig ../../data/post_data/$NAME/model_config/$TNAME/config.json --outpath ../../data/post_data/$NAME/model_config/$TNAME/embed_"$CKPT"_"$NAME".npz
fi

echo "Plotting"
cd $VIDA/vida/plot
python interact_plot.py --predata ../../data/post_data/$NAME/preprocess_"$NAME".npz --timedata ../../data/post_data/$NAME/time_"$NAME".npz --embeddata ../../data/post_data/$NAME/model_config/$TNAME/embed_"$CKPT"_"$NAME".npz --outpath ../../data/post_data/$NAME/model_config/$TNAME/plot_"$CKPT"_"$NAME"