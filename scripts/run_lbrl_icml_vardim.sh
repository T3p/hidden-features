#!/bin/zsh

TODAY=$(date +'%Y-%m-%d')
HORIZON=50000
SEEDS=3720889,11982979,8840435,11768055,1522722,6914388,9554994,6882166,11793768,8405117,1693841,2630410,10264872,7384624,12199513,10319645,9158576,1466036,3551766,9752720,8016258,2792641,51384,6236989,2655804,5701105,9032410,2139224,1239590,7058753,10774809,3216079,8717004,11337355,10844965,1693372,6551445,11142080,3791269,11279830,8009338,10498181,7071885,7686988,4954643,8176961,10066962,6898971,10539550,2640120,9585329,12214584,3859085,3383253,5016153,4815049,11032438,4544025,6563435,7919627,2473757,1559856,5286432,10606846,10306669,11679852,12019890,4055128,11070221,5387219,4119084,12088565,951829,1886485,5000482,8693874,8799800,10817657,9378413,7405943,1991024,5431925,5137938,10020314,9152913,3223,5061577,4481533,1943229,12216920,8977808,1485647,7972999,1025108,2168889,2317277,953921,3820345,1350207,8258964
for DOMAIN in vardim_icml_realizable vardim_icml_real_nohls vardim_icml_nonrealizable
do
    python new_lbrl_runner.py --multirun horizon=${HORIZON} domain=${DOMAIN}.yaml algo=srllinucb_mineig_norm,srllinucb_avg_quad_norm,srlegreedy_mineig_norm,srlegreedy_avg_quad_norm,leader seed=${SEEDS} hydra.sweep.dir=${DOMAIN}_${TODAY}/\${algo} use_wandb=false use_tb=true eps_decay=sqrt

    python new_lbrl_runner.py --multirun horizon=${HORIZON} domain=${DOMAIN}.yaml algo=linucb,egreedyglrt rep_idx=0,1,2,3,4,5 seed=${SEEDS} hydra.sweep.dir=${DOMAIN}_${TODAY}/\${algo}_\${rep_idx} use_wandb=false use_tb=true eps_decay=sqrt
done

for DOMAIN in vardim_icml_real_nohls
do
    python new_lbrl_runner.py --multirun horizon=${HORIZON} domain=${DOMAIN}.yaml algo=srllinucb_mineig_norm,srllinucb_avg_quad_norm,srlegreedy_mineig_norm,srlegreedy_avg_quad_norm,leader seed=${SEEDS} hydra.sweep.dir=${DOMAIN}_${TODAY}/\${algo} use_wandb=false use_tb=true 
done

for DOMAIN in vardim_icml_nonrealizable
do
    python new_lbrl_runner.py --multirun horizon=${HORIZON} domain=${DOMAIN}.yaml algo=srllinucb_mineig_norm,srllinucb_avg_quad_norm,srlegreedy_mineig_norm,srlegreedy_avg_quad_norm,leader seed=${SEEDS} hydra.sweep.dir=${DOMAIN}_${TODAY}/\${algo} use_wandb=false use_tb=true eps_decay=sqrt
done