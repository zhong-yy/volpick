program_dir=/home/zhongyiyuan/volpick/volpick/model
fractions=(
0.05
0.1
0.3
0.5
0.7
0.9
)

config_file=/home/zhongyiyuan/volpick/model_training/configs_testsize/p.json
for frac in ${fractions[@]};
do
    echo $frac
    python ${program_dir}/train.py --config ${config_file} --fraction ${frac}
done
