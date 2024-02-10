#!/bin/bash

echo "Train model from scratch. Be sure that you have already created the dataset.json file!"
read -p "Enter dataset id: " id
read -p "Enter train config (ex. 3d_fullres or custom 3d_fullres_mosaic, 3d_fullres_mosaic_resenc config): " conf
read -p "Enter train fold ([0-4] or all usually): " fold
read -p "Enter number of processes (1-20): " np
read -p "Enter trainer you would like to choose (nnUNetTrainer, nnUNetTrainerNoMirroring, nnUNetTrainer_Xepochs-where X could be in [100, 250, 2000, 4000, 8000], nnUNetTrainerDiceCELoss_noSmooth): " trainer

echo "Preparing dataset..."
nnUNetv2_plan_and_preprocess -d "$id" -c "$conf" --verify_dataset_integrity -np "$np"

echo "Dataset prepared. Launching the train..."
nnUNetv2_train "$id" --c "$conf" "$fold" -tr "$trainer" --npz