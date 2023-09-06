#!/bin/bash

echo "Train model from scratch. Be sure that you have already created the dataset.json file!"
read -p "Enter dataset id: " id
read -p "Enter train config (ex. 3d_fullres, 3d_lowers, 2d): " conf
read -p "Enter train fold ([0-4] or all usually): " fold
read -p "Enter number of processes (1-20): " np
read -p "Enter trainer you would like to choose (nnUNetTrainer, nnUNetTrainerNoMirroring, nnUNetTrainerNoDA, nnUNetTrainer4000epochs, nnUNetTrainer2000epochs, nnUNetTrainer500epochs, nnUNetTrainer250epochs, nnUNetTrainer150epochs): " trainer

echo "Preparing dataset..."
nnUNetv2_plan_and_preprocess -d "$id" -c "$conf" --verify_dataset_integrity -np "$np"

read -p "Do you want to customize training config? [y/n]: " is_new_conf
if [[ "$is_new_conf" == "Y" || "$is_new_conf" == "y" ]]; then
        echo "Creating custom config..."
        echo "WARNING! New config will set 1.5mm as default spacing value."
        read -p "Enter new batch size: " batch_size
        read -p "Enter nnUNetPlans file path: " conf_path
        python3 scripts/new_spacing.py "$conf_path" "$batch_size"
        conf="3d_fullres_custom"
        nnUNetv2_preprocess -d "$id" -c "$conf" --verify_dataset_integrity  -np "$np"
else
        echo "Cutstomization ignored."
fi

echo "Dataset prepared. Launching the train..."
nnUNetv2_train "$id" --c "$conf" "$fold" -tr "$trainer" --npz