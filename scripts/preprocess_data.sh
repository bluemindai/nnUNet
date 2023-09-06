#!/bin/bash

read -p "Enter dataset id : " id
read -p "Enter train config (ex. 3d_fullres, 3d_lowres, 2d) : " conf
read -p "Enter number of process (3 better) : " np

nnUNetv2_plan_and_preprocess -d $id -c $conf --verify_dataset_integrity -np $np