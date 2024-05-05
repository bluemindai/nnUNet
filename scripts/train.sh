#!/bin/bash

echo "Train model from scratch. Be sure that you have already created the dataset.json file!"
read -p "Enter dataset id: " id
read -p "Enter train config (ex. 3d_fullres): " conf
read -p "Enter train fold ([0-4] or all usually): " fold
read -p "Enter trainer you would like to choose (nnUNetTrainer, nnUNetTrainerNoMirroring, nnUNetTrainer_Xepochs-where X could be in [100, 250, 2000, 4000, 8000], nnUNetTrainerDiceCELoss_noSmooth): " trainer


echo "Launching the train..."
nnUNetv2_train "$id" --c "$conf" "$fold" -tr "$trainer"