#!/bin/bash

echo "Mosaic Preprocessor. Be sure you have a dataset.json file in the raw directory!"
read -p "Enter dataset id: " id
read -p "Enter resolution (low | std | high): " res
read -p "Enter resample (default | NoResample): " resample
read -p "Enter configuration (press Enter to set automatically): " config
read -p "Enter planner (press Enter to set automatically): " planner
read -p "Enter GPU memory (press Enter to set automatically): " gpu_mem
read -p "Enter number of processes (1-20, press Enter to use default 8): " np

# Устанавливаем количество процессов по умолчанию, если не указано
[[ -z "$np" ]] && np=8

declare -a spacing

# Настройка разрешения, планировщика, автоматическое определение памяти и конфигурации флага
case "$res" in
    low)
        echo -e "\nLow resolution selected. Setting spacing at 3.0 3.0 3.0"
        spacing=(3.0 3.0 3.0)
        [[ -z "$planner" ]] && planner="nnUNetPlannerResEncM"
        [[ -z "$gpu_mem" ]] && gpu_mem=12
        [[ -z "$config" ]] && config="3d_fullres_lowres_NoRsmp"
        ;;
    std)
        echo -e "\nStandard resolution selected. Setting spacing at 1.5 1.5 1.5"
        spacing=(1.5 1.5 1.5)
        [[ -z "$planner" ]] && planner="nnUNetPlannerResEncL"
        [[ -z "$gpu_mem" ]] && gpu_mem=24
        [[ -z "$config" ]] && config="3d_fullres_stdres_NoRsmp"
        ;;
    high)
        echo -e "\nHigh resolution selected. Setting spacing at 1.0 1.0 1.0"
        spacing=(1.0 1.0 1.0)
        [[ -z "$planner" ]] && planner="nnUNetPlannerResEncXL"
        [[ -z "$gpu_mem" ]] && gpu_mem=40
        [[ -z "$config" ]] && config="3d_fullres_highres_NoRsmp"
        ;;
    *)
        echo -e "\nInvalid resolution option."
        exit 1
        ;;
esac

# Обработка параметра resample
case "$resample" in
    default)
        echo -e "\nResample option 'default' selected."
        planner="${planner}"
        config="${config/_NoRsmp/}"
        ;;
    NoResample)
        echo -e "\nResample option 'NoResample' selected."
        ;;
    *)
        echo -e "\nInvalid resample option."
        exit 1
        ;;
esac

nnUNetv2_plan_and_preprocess -d "$id" \
    -c "$config" \
    -pl "$planner" \
    -overwrite_target_spacing "${spacing[@]}" \
    -gpu_memory_target "$gpu_mem" \
    -overwrite_plans_name "Mosaic_${planner}_${res}res_${resample}_${gpu_mem}G" \
    -np "$np" \
    --verify_dataset_integrity