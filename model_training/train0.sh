program_dir=/home/zhongyiyuan/volpick/volpick/model
config_dir=/home/zhongyiyuan/volpick/model_training/configs_tune
config_files=(
# p_1024_1e-03_ga10_400_s.json
# p_1024_1e-03_ga20_400_s.json
# e_1024_1e-04_ga10_400_s.json
# p_1024_1e-04_ga10_400_s.json
# p_1024_1e-04_ga20_400_s.json
# e_1024_5e-04_ga10_400_s.json
# p_512_1e-03_ga10_400_s.json
# e_512_1e-04_ga20_400_s.json
# p_512_5e-04_ga10_400_s.json
# e_512_5e-04_ga20_400_s.json
# p_512_5e-04_ga20_400_s.json
# e_1024_1e-03_ga20_400_s.json

# p_512_5e-04_ga20_600_s.json
# e_1024_1e-03_ga20_600_s.json
# e_512_5e-04_ga20_400_s_preinstance.json
# p_1024_1e-03_ga20_400_s_preinstance.json
# e_512_5e-04_ga20_600_s_preinstance.json
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
