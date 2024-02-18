#!/bin/bash
read -p "Enter dataset id : " id
read -p "Enter train config (ex. 3d_fullres, 3d_fullres_mosaic_arch2, 3d_fullres_mosaic_resenc) : " conf
read -p "Enter output path: " output

nnUNetv2_plot_overlay_pngs -d $id -o $output -c $conf
