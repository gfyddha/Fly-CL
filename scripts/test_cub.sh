#! /bin/bash

cd ..

python main.py --dataset CUB-200-2011 --num_classes 200 --num_tasks 10 --model_name vit_base_patch16_224 --embedding_dim 768 --expand_dim 10000 --synaptic_degree 300 --coding_level 0.3 --seed 2023 --batch_size 128 --gpu 1 --data_augmentation vit --ridge_lower 6 --ridge_upper 10