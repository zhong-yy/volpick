program_dir=/home/zhongyiyuan/volpick/volpick/model
config_dir=/home/zhongyiyuan/volpick/model_training/configs_tune
config_files=(
e_1024_1e-03_ga20_400_s_preinstance.json
)

echo "The following files will be processed"
for config_file in ${config_files[@]};
do
    echo $config_file
done

echo "Start training ..."
for config_file in ${config_files[@]};
do
    echo $config_file
    python ${program_dir}/train.py --config ${config_dir}/${config_file}
done
