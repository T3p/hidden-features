#!/bin/zsh

TODAY=$(date +'%Y-%m-%d')
HORIZON=50000
SEEDS=3072810,4961752,6429502,10459894,12082893,3278453,9512236,8456663,10318982,6442460,7669135,7335931,9585688,4692096,649642,7309948,5022585,6452796,6622844,5555361,5150257,7302292,10384878,11726365,9568168,6034637,3179151,11584587,11135058,6379609,724596,11930566,7232031,5035706,11396216,12125598,1293540,1481462,3157563,11618527,4811025,4567359,5134366,4074432,592797,1098404,4937790,2811825,10062358,8725858
for DOMAIN in vardim_icml_realizable vardim_icml_real_nohls vardim_icml_nonrealizable
do
    python new_lbrl_runner.py --multirun horizon=${HORIZON} domain=${DOMAIN}.yaml algo=srllinucb_mineig_norm,srllinucb_avg_quad_norm,srlegreedy_mineig_norm,srlegreedy_avg_quad_norm,leader seed=${SEEDS} hydra.sweep.dir=${DOMAIN}_${TODAY}/\${algo} use_wandb=true use_tb=true exp_name=${DOMAIN}_${TODAY}_\${algo}

    python new_lbrl_runner.py --multirun horizon=${HORIZON} domain=${DOMAIN}.yaml algo=linucb,egreedyglrt rep_idx=0,1,2,3,4,5 seed=${SEEDS} hydra.sweep.dir=${DOMAIN}_${TODAY}/\${algo}_\${rep_idx} use_wandb=true use_tb=true exp_name=${DOMAIN}_${TODAY}_\${algo}_\${rep_idx}
done
