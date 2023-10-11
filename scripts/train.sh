#!/bin/bash

echo "Train model from scratch. Be sure that you have already created the dataset.json file!"
read -p "Enter dataset id: " id
read -p "Enter train config (ex. 3d_fullres, 3d_lowers, 2d or custom 3d_fullres_mosaic config): " conf
read -p "Enter train fold ([0-4] or all usually): " fold
read -p "Enter number of processes (1-20): " np
read -p "Enter trainer you would like to choose (nnUNetTrainer, nnUNetTrainerNoMirroring, nnUNetTrainer_Xepochs-where X could be in [100, 250, 2000, 4000, 8000], nnUNetTrainerDiceCELoss_noSmooth): " trainer

echo "Preparing dataset..."
nnUNetv2_plan_and_preprocess -d "$id" -c "$conf" --verify_dataset_integrity -np "$np"

echo "Dataset prepared. Launching the train..."
nnUNetv2_train "$id" --c "$conf" "$fold" -tr "$trainer" --npz

# nnUNetv2_train 901 3d_fullres 0 -tr nnUNetTrainerDiceCELoss_noSmooth_150epochs -pretrained_weights /media/eolika/8224BB4D24BB42C9/BLUEMIND/nnUNet/trains/nnUNet_results/Dataset002_mosaicV1/nnUNetTrainerDiceCELoss_noSmooth_250epochs__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth --npz
# nnUNetv2_train 101 3d_fullres_mosaic 0 -tr nnUNetTrainerNoMirroring --npz