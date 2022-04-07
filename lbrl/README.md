    python lbrl_runner.py algo=leaderselect

The output is saved in lbrl_output with date and timestamp


It is also possible to run multiple repetitions by using the following command

    python lbrl_runner.py algo=leaderselect seed=1,2,3,4

The output is saved in lbrl_multirun with a subfolder for each configuration.

With the standard configuration we can run

    python lbrl_runner.py --multirun algo=leaderselect normalize_mineig=True,False seed=1,2,3,4,5,6,7,8,9,10
    python lbrl_runner.py --multirun algo=leader seed=1,2,3,4,5,6,7,8,9,10 
    python lbrl_runner.py --multirun algo=linucb linucb_rep=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 seed=1,2,3,4,5,6,7,8,9,10 

To parse the result you can use the script `lbrl_runner_parse_results.ipynb`