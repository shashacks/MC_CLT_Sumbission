#!/bin/bash
# ex ./training_command_mc_clt_model.sh Hopper-v3 0.01 20
# python -m spinup.run mc_clt_ppo --hid "[64,32]" --env Hopper-v3 --exp_name test2 --seed 5000
env=$1
alpha=$2
mn_std=$3

for i in {0..1}
do
    python -m spinup.run drl_lclt_ppo --hid "[64,32]" --env ${1} --exp_name ${1}_a${2}_m${3}_MC_CLT_model --seed 500${i} --alpha ${2} --mn_std ${3}
done

