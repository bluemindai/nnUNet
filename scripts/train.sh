#!/bin/bash

echo "Train model from scratch. Be sure that you have already created the dataset.json file!"
read -p "Enter dataset id: " id
echo "Select train config:
0 - 3d_fullres
1 - 3d_fullres_lowres_NoRsmp
2 - 3d_fullres_stdres_NoRsmp
3 - 3d_fullres_highres_NoRsmp"
read -p "Enter train config number (Press enter to set default 0 - 3d_fullres): " conf_num
echo "Select trainer:
0 - nnUNetTrainer
1 - nnUNetTrainerNoMirroring
2 - nnUNetTrainer_250epochs_NoMirroring
3 - nnUNetTrainer_500epochs_NoMirroring
4 - nnUNetTrainer_750epochs_NoMirroring
5 - nnUNetTrainer_2000epochs_NoMirroring
6 - nnUNetTrainer_4000epochs_NoMirroring
7 - nnUNetTrainer_8000epochs_NoMirroring"
read -p "Enter trainer number: " trainer_num
read -p "Enter train fold ([0-4] or all usually): " fold

# Устанавливаем значение по умолчанию для конфигурации и плана, если они не указаны
case "$conf_num" in
    0) conf="3d_fullres" ;;
    1) conf="3d_fullres_lowres_NoRsmp" ;;
    2) conf="3d_fullres_stdres_NoRsmp" ;;
    3) conf="3d_fullres_highres_NoRsmp" ;;
    *) conf="3d_fullres" ;;
esac

plans="nnUNetPlans"

case "$trainer_num" in
    0) trainer="nnUNetTrainer" ;;
    1) trainer="nnUNetTrainerNoMirroring" ;;
    2) trainer="nnUNetTrainer_250epochs_NoMirroring" ;;
    3) trainer="nnUNetTrainer_500epochs_NoMirroring" ;;
    4) trainer="nnUNetTrainer_750epochs_NoMirroring" ;;
    5) trainer="nnUNetTrainer_2000epochs_NoMirroring" ;;
    6) trainer="nnUNetTrainer_4000epochs_NoMirroring" ;;
    7) trainer="nnUNetTrainer_8000epochs_NoMirroring" ;;
    *) echo "Error: Invalid trainer number."; exit 1 ;;
esac

# Проверяем, указаны ли fold и trainer, и выводим ошибку в противном случае
if [[ -z "$fold" ]]; then
    echo "Error: Train fold is required."
    exit 1
fi

echo "Launching the training process..."
nnUNetv2_train "$id" "$conf" "$fold" -p "$plans" -tr "$trainer"