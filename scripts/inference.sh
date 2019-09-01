#~/bin/bash


python inference.py \
    --dataDir ./data \
    --cls  modelNet40_2.5k \
    --model ./out/log1/best_net__.pth \
    --outf ./out/log1 \
    --type test \
    --num_points 2500 \
    --chamLoss_wt \
