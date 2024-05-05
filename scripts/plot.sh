#!/bin/bash
read -p "Enter dataset id: " id
read -p "Enter train config (Press enter to set default 3d_fullres): " conf
read -p "Enter plans identifier (Press enter to set default nnUNetPlans): " p
read -p "Enter output base path: " output_base

# Устанавливаем значение по умолчанию для конфигурации, если оно не указано
[[ -z "$conf" ]] && conf="3d_fullres"
# Устанавливаем значение по умолчанию для планов, если оно не указано
[[ -z "$p" ]] && p="nnUNetPlans"

# Создаем полный путь для директории plots
output="${output_base}/plots"

# Создаем директорию, если она не существует, с проверкой прав доступа
mkdir -p "$output" || { echo "Error: Unable to create directory '$output'."; exit 1; }

# Запуск команды с правильным путем
nnUNetv2_plot_overlay_pngs -d "$id" -o "$output" -p "$p" -c "$conf"