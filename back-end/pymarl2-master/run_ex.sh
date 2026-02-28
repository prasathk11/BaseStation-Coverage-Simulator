#!/bin/bash

# 定义epsilon值列表
epsilons=("0" "0.2" "0.4" "0.6" "0.8")

# 遍历每个epsilon值
for epsilon in "${epsilons[@]}"; do

    # 使用find查找包含该epsilon值的模型文件夹路径
    model_paths=$(find results/models -type d -name "vdn_env=dt_rb_allocation_10u_eps_${epsilon}*")

    ex_count = 1

    # 遍历找到的每个模型路径
    for model_path in $model_paths; do
        # 提取模型文件夹名
        model_folder=$(basename "$model_path")
        
        # 提取时间戳部分
        timestamp=$(echo "$model_folder" | awk -F '__' '{print $2}')

        # 生成符合要求的name
        name="vdn_env=dt_rb_allocation_10u_eps_${epsilon}_${exp_count}_${timestamp}"

        # 修改default.yaml的path参数为当前模型路径
        sed -i "s|^checkpoint_path:.*|checkpoint_path: ${model_path}|" src/config/default.yaml
        sed -i "s|^name:.*|name: ${name}|" src/config/algo/vdn.yaml
        sed -i "s|^name:.*|name: ${name}|" src/config/algo/vdn.yaml

        # 运行实验
        python3 src/main.py --env-config= &
        ((exp_count++))
    done
done

wait  # 等待所有实验完成