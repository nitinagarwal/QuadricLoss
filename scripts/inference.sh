#~/bin/bash


python inference.py \
    --dataDir ./data \
    --cls abc_2.5k \
    --model ./log/log1 \
    --outf out \
    --type test \
    --num_points 2500 \

