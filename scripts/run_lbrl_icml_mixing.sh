#!/bin/zsh

TODAY=$(date +'%Y-%m-%d')
HORIZON=80000
SEEDS=523638177,2041280004,1414771290,631534312,1453736814,1397631146,2638535594,1496331968,2726184843,4051158523,3527264775,3958859807,2475161690,948634907,498893635,4011596387,1928773751,2463789121,1355946980,582057396,897038420,677529307,4251148658,1812338824,3650947947,1008507280,3791627088,1505862441,3760189604,4281275324,4256252563,3244565067,2743459145,1003717380,1927117284,3081151930,3321129299,3173403295,4269545983,72630365,2370437039,1961989697,2114798954,2948483590,3035586253,2605854944,3557811144,2428106631,4002200162,276945325
for DOMAIN in mixing
do
    # srllinucb_mineig_norm,srllinucb_minfeat_norm_adaptive,srllinucb_minfeat_norm,srlegreedy_mineig_norm,srlegreedy_minfeat_norm_adaptive,srlegreedy_minfeat_norm,leader
    python new_lbrl_runner.py --multirun horizon=${HORIZON} domain=${DOMAIN}.yaml algo=srllinucb_mineig_norm,srllinucb_minfeat_norm_adaptive,srllinucb_minfeat_norm,srlegreedy_mineig_norm,srlegreedy_minfeat_norm_adaptive,srlegreedy_minfeat_norm,leader seed=${SEEDS} hydra.sweep.dir=new_${DOMAIN}_${TODAY}/\${algo} use_wandb=false use_tb=true eps_decay=cbrt glrt_scale=1 forcedexp=false reg_matrix_rl=false

    # python new_lbrl_runner.py --multirun horizon=${HORIZON} domain=${DOMAIN}.yaml algo=linucb,egreedyglrt rep_idx=0,1,2,3,4,5 seed=${SEEDS} hydra.sweep.dir=${DOMAIN}_${TODAY}/\${algo}_\${rep_idx} use_wandb=false use_tb=true eps_decay=cbrt
done
