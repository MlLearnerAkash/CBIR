#!/bin/bash

python generate_DB.py \
    --tag debug \
    --cfg configs/swin_tiny_patch4_window7_224.yaml \
    --batch-size 1 \
    --resume checkpoints/swin_tiny_patch4_window7_224.pth \
    --data-path database/data\
    --local_rank 1
