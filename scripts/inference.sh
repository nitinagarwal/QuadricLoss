#~/bin/bash


python inference.py \
    --dataDir ../../data/all_data \
    --cls  modelNet40_2.5k \
    --model ./out/log30/best_net_201.pth \
    --outf ./out/log30 \
    --type test \
    --num_points 2500 \
    --quadLoss_wt \
