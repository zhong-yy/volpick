program_dir=/home/zhongyiyuan/volpick/volpick/model
config_dir=/home/zhongyiyuan/volpick/model_training/configs_swa
config_file=p.json
echo "Start training ..."
python ${program_dir}/train.py --config ${config_dir}/${config_file}
