#!/bin/bash

echo "Train model from scratch. Be sure that you have already created the dataset.json file!"
read -p "Enter dataset id: " id
read -p "Enter train config (ex. 3d_fullres or custom 3d_fullres_mosaic, 3d_fullres_mosaic_resenc, 3d_fullres_mosaic_resenc_192x192x192_bs2_1mm config): " conf
read -p "Enter train fold ([0-4] or all usually): " fold
read -p "Enter trainer you would like to choose (nnUNetTrainer, nnUNetTrainerNoMirroring, nnUNetTrainer_Xepochs-where X could be in [100, 250, 2000, 4000, 8000], nnUNetTrainerDiceCELoss_noSmooth): " trainer


echo "Launching the train..."
nnUNetv2_train "$id" --c "$conf" "$fold" -tr "$trainer"

# nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_mosaic_resenc_192x192x192_bs3_1mm 0 -num_gpus 8