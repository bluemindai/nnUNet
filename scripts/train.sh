#!/bin/bash

echo "Train model from scratch. Be sure that you have already created the dataset.json file!"
read -p "Enter dataset id: " id
read -p "Enter train config (Press enter to set default 3d_fullres): " conf
read -p "Enter plans identifier (Press enter to set  nnUNetPlans): " p
read -p "Enter train fold ([0-4] or all usually): " fold
read -p "Enter trainer (nnUNetTrainer, nnUNetTrainerNoMirroring, nnUNetTrainer_Xepochs_NoMirroring etc.): " trainer

# Устанавливаем значение по умолчанию для конфигурации и плана, если они не указаны
[[ -z "$conf" ]] && conf="3d_fullres"
[[ -z "$p" ]] && p="nnUNetPlans"

# Проверяем, указаны ли fold и trainer, и выводим ошибку в противном случае
if [[ -z "$fold" ]]; then
    echo "Error: Train fold is required."
    exit 1
fi

if [[ -z "$trainer" ]]; then
    echo "Error: Trainer is required."
    exit 1
fi

echo "Launching the training process..."
nnUNetv2_train "$id" "$conf" "$fold" -p "$p" -tr "$trainer"