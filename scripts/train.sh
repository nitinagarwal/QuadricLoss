#~/bin/bash

# training the network

python train.py \
    --dataDir /home/minions/Dropbox/GraphicsLab/Projects/3D_Content_Creation/data/all_data \
    --num_points 2500 \
    --cls abc_2.5k \
    --batchSize 8 \
    --nepoch 300 \
    --logf log1 \
    --chamLoss_wt 0.0 \
    --quadLoss_wt 1.0 \
    --sufNorLoss_wt 0.0 \
    --sufLoss_wt 0.0 \
    --lr 0.0001 \
    --lr_decay 0.8 \
    --lr_step_size 100 \

