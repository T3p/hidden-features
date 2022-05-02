# Learning Representations for Contextual Bandits


Do not look into tmp_code

### Running experiments in XBRL (ie deep)

    python run_multiclass.py --dataset magic --bandittype extended --algo nnlinucb  --max_epochs 50 --update_every 500 --lr 0.001 --batch_size 256 --config_name _incremental


### Running experiments in LBRL (ie linear problem)

    python lbrl_runner.py --multirun horizon=100000 domain=vardimtest.yaml algo=leader,leaderselectlb seed=3072810,4961752,6429502,10459894,12082893,3278453,9512236,8456663,10318982,6442460,7669135,7335931,9585688,4692096,649642,7309948,5022585,6452796,6622844,5555361 hydra.sweep.dir=vardimtest_\${now:%Y-%m-%d}/\${algo}

    python lbrl_runner.py --multirun horizon=100000 domain=vardimtest.yaml algo=leaderselect normalize_mineig=true,false seed=3072810,4961752,6429502,10459894,12082893,3278453,9512236,8456663,10318982,6442460,7669135,7335931,9585688,4692096,649642,7309948,5022585,6452796,6622844,5555361 hydra.sweep.dir=vardimtest_\${now:%Y-%m-%d}/\${algo}_norm-\${normalize_mineig}

    python lbrl_runner.py --multirun horizon=100000 domain=vardimtest.yaml algo=linucb linucb_rep=0,1,2,3,4 seed=3072810,4961752,6429502,10459894,12082893,3278453,9512236,8456663,10318982,6442460,7669135,7335931,9585688,4692096,649642,7309948,5022585,6452796,6622844,5555361 hydra.sweep.dir=vardimtest_\${now:%Y-%m-%d}/\${algo}_\${linucb_rep}
