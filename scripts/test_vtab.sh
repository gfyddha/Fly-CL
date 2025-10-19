#! /bin/bash

cd ..

python main.py --dataset VTAB --num_classes 50 --num_tasks 5 --model_name vit_base_patch16_224 --embedding_dim 768 --expand_dim 10000 --synaptic_degree 300 --coding_level 0.3 --seed 2023 --batch_size 128 --gpu 6 --data_augmentation vit --ridge_lower 6 --ridge_upper 10