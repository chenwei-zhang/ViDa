#!/bin/bash

VIDA="/Users/chenwei/Desktop/Github/ViDa"
NAME='central_toehold8'

### PLOT ###
TNAME='25-0531-1435'
CKPT='checkpoint_epoch_9'

echo "Embedding"
cd $VIDA/vida/models
if [ -f "$VIDA/data/post_data/$NAME/model_config/$TNAME/embed_"$CKPT"_"$NAME".npz" ]; then
    echo "Embedding already exists, skip embedding"
else
    python embed_vida.py --data $VIDA/data/post_data/$NAME/dataloader_"$NAME".pkl.gz --model $VIDA/data/post_data/$NAME/model_config/$TNAME/$CKPT.pt --fconfig $VIDA/data/post_data/$NAME/model_config/$TNAME/config.json --outpath $VIDA/data/post_data/$NAME/model_config/$TNAME/embed_"$CKPT"_"$NAME".npz
fi

echo "Plotting"
cd $VIDA/vida/plot
python interact_plot.py --predata $VIDA/data/post_data/$NAME/preprocess_"$NAME".npz --timedata $VIDA/data/post_data/$NAME/time_"$NAME".npz --embeddata $VIDA/data/post_data/$NAME/model_config/$TNAME/embed_"$CKPT"_"$NAME".npz --outpath $VIDA/data/post_data/$NAME/model_config/$TNAME/plot_"$CKPT"_"$NAME"