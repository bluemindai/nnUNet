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

echo "Extract source dataset."
nnUNetv2_extract_fingerprint -d "$src"
echo "Move plans between datasets."
nnUNetv2_move_plans_between_datasets -s "$src" -t "$trg" -sp nnUNetPlans -tp nnUNetPlans
echo "Train..."
nnUNetv2_train "$trg" "$config" "$fold" -pretrained_weights $weights -tr "$trainer" --npz