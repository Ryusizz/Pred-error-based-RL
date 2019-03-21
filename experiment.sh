#!/bin/bash
declare -a args=("exp_name" "RConErr_idf_fromDyn_100M_dynloss00"
                 "env" "SeaquestNoFrameskip-v4"
                 "env_kind" "atari"
                 "policy_mode" "rnnerr"
                 "feat_learning" "idf"
                 "dyn_from_pixels" 0
                 "feat_sharedWpol" 0
                 "full_tensorboard_log" 1
                 "tboard_period" 10
                 "ext_coeff" 1.
                 "int_coeff" 0.
                 "save_dynamics" 0
                 "load_dir" "/result/SeaquestNoFrameskip-v4/RConErr_idf_forDyn_100M"
                 "num_timesteps" 200000000
                 "dyn_coeff" 0.
                 "aux_coeff" 0.
                 )

arraylength=${#args[@]}
argline=""
for (( i=0; i<${arraylength}; i=i+2 ));
do
    arg="--"${args[$i]}" "${args[$i+1]}" "
    argline="$argline""$arg"
done

mpiexec --allow-run-as-root -n 4 python3 rnn_run.py $argline


#env="SeaquestNoFrameskip-v4"
#env_kind="atari"
#policy_mode="rnnerr"
#feat_learning="idf"
#dyn_from_pixels=0
#feat_sharedWpol=0
#full_tensorboard_log=1
#tboard_period=10
#ext_coeff=1.
#int_coeff=0.
#save_dynamics=1
#load_dir=None
#num_timesteps=100000000

#tag="RConErr_idf_forDyn_100M"
#mpiexec --allow-run-as-root -n 4 python3 rnn_run.py --exp_name $tag --env $env --env_kind=$env_kind --ext_coeff=$ext_coeff --int_coeff=$int_coeff --policy_mode=$policy_mode --full_tensorboard_log=$full_tensorboard_log --tboard_period=$tboard_period --dyn_from_pixels=$dyn_from_pixels --feat_learning=$feat_learning --feat_sharedWpol=$feat_sharedWpol --save_dynamics=$save_dynamics --num_timesteps=$num_timesteps

