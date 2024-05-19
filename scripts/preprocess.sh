#!/bin/bash

echo "Mosaic Preprocessor. Be sure you have a dataset.json file in the raw directory!"
read -p "Enter dataset id: " id
read -p "Enter resolution (low | std | high): " res
read -p "Enter planner (press Enter to set automatically): " planner
read -p "Enter GPU memory (press Enter to set automatically): " gpu_mem
read -p "Enter number of processes (1-20, press Enter to use default 4): " np

# Устанавливаем количество процессов по умолчанию, если не указано
[[ -z "$np" ]] && np=4

declare -a spacing

# Настройка разрешения, планировщика и автоматическое определение памяти
case "$res" in
    low)
        echo -e "\nLow resolution selected. Setting spacing at 3.0 3.0 3.0"
        spacing=(3.0 3.0 3.0)
        [[ -z "$planner" ]] && planner="nnUNetPlannerResEncM"
        [[ -z "$gpu_mem" ]] && gpu_mem=12
        ;;
    std)
        echo -e "\nStandard resolution selected. Setting spacing at 1.5 1.5 1.5"
        spacing=(1.5 1.5 1.5)
        [[ -z "$planner" ]] && planner="nnUNetPlannerResEncL"
        [[ -z "$gpu_mem" ]] && gpu_mem=24
        ;;
    high)
        echo -e "\nHigh resolution selected. Setting spacing at 1.0 1.0 1.0"
        spacing=(1.0 1.0 1.0)
        [[ -z "$planner" ]] && planner="nnUNetPlannerResEncXL"
        [[ -z "$gpu_mem" ]] && gpu_mem=40
        ;;
    *)
        echo -e "\nInvalid resolution option."
        exit 1
        ;;
esac

nnUNetv2_plan_and_preprocess -d "$id" \
    -c 3d_fullres \
    -pl "$planner" \
    -overwrite_target_spacing "${spacing[@]}" \
    -gpu_memory_target "$gpu_mem" \
    -overwrite_plans_name "Mosaic_${planner}_${res}res_NoRsmp_${gpu_mem}G" \
    -np "$np" \
    #--verify_dataset_integrity