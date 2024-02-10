#!/bin/bash

echo "Prepare target dataset on pretrained set."
read -p "Enter source dataset: " id
read -p "Enter train config (ex. 3d_fullres, 3d_fullres_mosaic_resenc, 3d_fullres_mosaic_resenc_192x192x192_bs3_1mm): " conf
read -p "Enter a number of processors to use: " np

nnUNetv2_extract_fingerprint -d $id
nnUNetv2_plan_experiment -d $id
nnUNetv2_plan_experiment -d $id -pl ResEncUNetPlanner
nnUNetv2_preprocess -d $id -c $conf -np $np