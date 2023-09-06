#!/bin/bash

echo "Prepare target dataset on pretrained set."
read -p "Enter source dataset: " src
read -p "Enter target dataset: " trg
read -p "Enter nnUNet configuration (2d, 3d_fullres, 3d_lowres or custom config): " config
read -p "Enter number of processes (1-20 but 18 is well enough): " np
read -p "Path to weights: " weights
read -p "Enter train fold (0 usually): " fold
read -p "Enter trainer you would like to choose (nnUNetTrainer, nnUNetTrainerNoMirroring, nnUNetTrainerNoDA, nnUNetTrainer500epochs): " trainer

echo "Preprocess target dataset."
nnUNetv2_plan_and_preprocess -d "$trg" -c "$config" --verify_dataset_integrity -np "$np"

read -p "Do you want to customize training config? [y/n]: " is_new_conf
if [[ "$is_new_conf" == "Y" || "$is_new_conf" == "y" ]]; then
        echo "Creating custom config..."
        echo "WARNING! New config will set 1.5mm as default spacing value."
        read -p "Enter new batch size: " batch_size
        read -p "Enter nnUNetPlans file path: " conf_path
        python3 scripts/new_spacing.py "$conf_path" "$batch_size"
        conf="3d_fullres_custom"
        nnUNetv2_preprocess -d "$id" -c "$conf" -np "$np"
else
        echo "Cutstomization ignored."
fi

echo "Extract source dataset."
nnUNetv2_extract_fingerprint -d "$src"
echo "Move plans between datasets."
nnUNetv2_move_plans_between_datasets -s "$src" -t "$trg" -sp nnUNetPlans -tp nnUNetPlans
echo "Train..."
nnUNetv2_train "$trg" "$config" "$fold" -pretrained_weights $weights -tr "$trainer" --npz