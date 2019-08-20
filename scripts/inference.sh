#~/bin/bash


python inference.py \
    --dataDir /home/minions/Dropbox/GraphicsLab/Projects/3D_Content_Creation/data/all_data \
    --cls abc_2.5k \
    --model ./log/log1 \
    --outf out \
    --type test \
    --num_points 2500 \

