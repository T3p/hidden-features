# Learning Representations for Contextual Bandits


Do not look into tmp_code

# How to Replicate the Experiments

For the experiments in the main paper, you can run the following commands

### wheel

    python run_linearfinite.py -m domain=wheel.yaml horizon=200000 algo=nnlinucb check_glrt=True epsilon_decay=none bonus_scale=3 glrt_scale=5 layers=\"100,100,50,20,10\" weight_mse=1 weight_rayleigh=0 weight_min_features=0,1 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=713,464,777,879,660,608,773,919,591,229 use_tb=true use_maxnorm=False hydra.sweep.dir=expmain/wheel/ hydra.sweep.subdir=\${algo}_weak\${weight_min_features}_\${seed} hydra.launcher.submitit_folder=expmain/wheel/.slurm


    python run_linearfinite.py -m domain=wheel.yaml horizon=200000 algo=nnegreedy check_glrt=True epsilon_decay=cbrt bonus_scale=3 glrt_scale=5 layers=\"100,100,50,20,10\" weight_mse=1 weight_rayleigh=0 weight_min_features=0,1 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=713,464,777,879,660,608,773,919,591,229 use_tb=true use_maxnorm=False hydra.sweep.dir=expmain/wheel/ hydra.sweep.subdir=\${algo}_weak\${weight_min_features}_\${seed} hydra.launcher.submitit_folder=expmain/wheel/.slurm


# statlog

    python run_linearfinite.py -m domain=statlog.yaml horizon=500000 algo=nnlinucb check_glrt=True epsilon_decay=none bonus_scale=2 glrt_scale=3 layers=\"100,100,50,20,10\" weight_mse=1 weight_rayleigh=0 weight_min_features=0,1 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=713,464,777,879,660,608,773,919,591,229 use_tb=true use_maxnorm=False hydra.sweep.dir=expmain/statlog/ hydra.sweep.subdir=\${algo}_weak\${weight_min_features}_bs\${bonus_scale}_glrts\${glrt_scale}_\${seed} hydra.launcher.submitit_folder=expmain/statlog/.slurm

    python run_linearfinite.py -m domain=statlog.yaml horizon=200000 algo=nnegreedy check_glrt=True epsilon_decay=cbrt,sqrt bonus_scale=2 glrt_scale=3 layers=\"100,100,50,20,10\" weight_mse=1 weight_rayleigh=0 weight_min_features=0,1 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=713,464,777,879,660,608,773,919,591,229 use_tb=true use_maxnorm=False hydra.sweep.dir=expmain/statlog/ hydra.sweep.subdir=\${algo}_weak\${weight_min_features}_eps\${epsilon_decay}_glrts\${glrt_scale}_\${seed} hydra.launcher.submitit_folder=expmain/statlog/.slurm


# statlog NEW ABLATION
python run_linearfinite.py -m domain=statlog.yaml horizon=300000 algo=nnegreedy check_glrt=True epsilon_decay=cbrt,sqrt,frt glrt_scale=0,0.1,0.5,1,3 layers=\"100,100,50,20\",\"100,100,50,20,10\",\"100,100\",\"50,50,50,50\" weight_mse=1 weight_rayleigh=0 weight_min_features=0,1 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=713,464,777,879,660,608,773,919,591,229 use_tb=true use_maxnorm=False hydra.sweep.dir=ablation/statlog/run1 hydra.launcher.submitit_folder=ablation/statlog/run1/.slurm
python run_linearfinite.py -m domain=statlog.yaml horizon=300000 algo=nnegreedy check_glrt=True epsilon_decay=cbrt,sqrt,frt glrt_scale=0,0.1,0.5,1,3 layers=\"100,100,50,20\",\"100,100,50,20,10\",\"100,100\",\"50,50,50,50\" weight_mse=1 weight_rayleigh=1 weight_min_features=0 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=713,464,777,879,660,608,773,919,591,229 use_tb=true use_maxnorm=False hydra.sweep.dir=ablation/statlog/run2 hydra.launcher.submitit_folder=ablation/statlog/run2/.slurm


# last set of params

python run_linearfinite.py -m domain=statlog.yaml algo=nnegreedy epsilon_decay=frt check_glrt=true layers=\"50,50,50,50,10\" horizon=200000 weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=1 hydra.sweep.dir=ablation/statlog_50/run1 hydra.launcher.submitit_folder=ablation/statlog_50/run1/.slurm

python run_linearfinite.py -m domain=statlog.yaml algo=nnlinucb epsilon_decay=none check_glrt=true layers=\"50,50,50,50,10\" horizon=200000 weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=1 glrt_scale=1 hydra.sweep.dir=ablation/statlog_50/run2 hydra.launcher.submitit_folder=ablation/statlog_50/run2/.slurm
