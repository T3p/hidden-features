#!/bin/zsh

TODAY=$(date +'%Y-%m-%d')
HORIZON=80000
SEEDS=3720889,11982979,8840435,11768055,1522722,6914388,9554994,6882166,11793768,8405117,1693841,2630410,10264872,7384624,12199513,10319645,9158576,1466036,3551766,9752720,8016258,2792641,51384,6236989,2655804,5701105,9032410,2139224,1239590,7058753,10774809,3216079,8717004,11337355,10844965,1693372,6551445,11142080,3791269,11279830,8009338,10498181,7071885,7686988,4954643,8176961,10066962,6898971,10539550,2640120
DOMAIN=vardim_weakhls
# vardim_icml_nonrealizable
# vardim_icml_realizable
# vardim_icml_real_nohls

python new_lbrl_runner.py --multirun horizon=${HORIZON} domain=${DOMAIN}.yaml algo=leader seed=${SEEDS} hydra.sweep.dir=new_${DOMAIN}_${TODAY}/\${algo} use_wandb=false use_tb=true eps_decay=cbrt glrt_scale=1 forcedexp=false reg_matrix_rl=false
python new_lbrl_runner.py --multirun horizon=${HORIZON} domain=${DOMAIN}.yaml algo=srllinucb_minfeat_norm seed=${SEEDS} hydra.sweep.dir=new_${DOMAIN}_${TODAY}/\${algo} use_wandb=false use_tb=true eps_decay=cbrt glrt_scale=1 forcedexp=false reg_matrix_rl=false
python new_lbrl_runner.py --multirun horizon=${HORIZON} domain=${DOMAIN}.yaml algo=srlegreedy_minfeat_norm seed=${SEEDS} hydra.sweep.dir=new_${DOMAIN}_${TODAY}/\${algo} use_wandb=false use_tb=true eps_decay=cbrt glrt_scale=1 forcedexp=false reg_matrix_rl=false
python new_lbrl_runner.py --multirun horizon=${HORIZON} domain=${DOMAIN}.yaml algo=srllinucb_mineig_norm seed=${SEEDS} hydra.sweep.dir=new_${DOMAIN}_${TODAY}/\${algo} use_wandb=false use_tb=true eps_decay=cbrt glrt_scale=1 forcedexp=false reg_matrix_rl=false
python new_lbrl_runner.py --multirun horizon=${HORIZON} domain=${DOMAIN}.yaml algo=srlegreedy_mineig_norm seed=${SEEDS} hydra.sweep.dir=new_${DOMAIN}_${TODAY}/\${algo} use_wandb=false use_tb=true eps_decay=cbrt glrt_scale=1 forcedexp=false reg_matrix_rl=false
