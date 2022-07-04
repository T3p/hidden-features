# Learning Representations for Contextual Bandits


Do not look into tmp_code

# How to Replicate the Experiments

For the experiments in the main paper, you can run the following commands

### wheel

    python run_linearfinite.py -m domain=wheel.yaml domain.ncontexts=100 algo=nnegreedy epsilon_decay=cbrt,sqrt check_glrt=true layers=\"50,50,50,50,10\" horizon=500000 weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=1,2,5 USE_MAXNORM=FALSE hydra.sweep.dir=expablation/wheel100_5010/run1 hydra.launcher.submitit_folder=expablation/wheel100_5010/run1/.slurm

    python run_linearfinite.py -m domain=wheel.yaml domain.ncontexts=100 algo=nnlinucb epsilon_decay=none check_glrt=true layers=\"50,50,50,50,10\" horizon=500000 weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=1,2 glrt_scale=1,2,5 USE_MAXNORM=FALSE hydra.sweep.dir=expablation/wheel100_5010/run2 hydra.launcher.submitit_folder=expablation/wheel100_5010/run2/.slurm

    python run_linearfinite.py -m domain=wheel.yaml domain.ncontexts=100 algo=nnegreedy epsilon_decay=cbrt,sqrt check_glrt=false layers=\"50,50,50,50,10\" horizon=500000 weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=1 hydra.sweep.dir=expablation/wheel100_5010/run3 hydra.launcher.submitit_folder=expablation/wheel100_5010/run3/.slurm

    python run_linearfinite.py -m domain=wheel.yaml domain.ncontexts=100 algo=nnlinucb epsilon_decay=none check_glrt=false layers=\"50,50,50,50,10\" horizon=500000 weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=1,2 glrt_scale=1 hydra.sweep.dir=expablation/wheel100_5010/run4 hydra.launcher.submitit_folder=expablation/wheel100_5010/run4/.slurm


# statlog, mushroom, magic

    python run_linearfinite.py -m domain=statlog.yaml algo=nnegreedy epsilon_decay=cbrt,sqrt check_glrt=true layers=\"50,50,50,50,10\" horizon=500000 weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=1,2,5 USE_MAXNORM=FALSE hydra.sweep.dir=expablation/statlog_5010/run1 hydra.launcher.submitit_folder=expablation/statlog_5010/run1/.slurm

    python run_linearfinite.py -m domain=statlog.yaml algo=nnlinucb epsilon_decay=none check_glrt=true layers=\"50,50,50,50,10\" horizon=500000 weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=1,2 glrt_scale=1,2,5 USE_MAXNORM=FALSE hydra.sweep.dir=expablation/statlog_5010/run2 hydra.launcher.submitit_folder=expablation/statlog_5010/run2/.slurm

    python run_linearfinite.py -m domain=statlog.yaml algo=nnegreedy epsilon_decay=cbrt,sqrt check_glrt=false layers=\"50,50,50,50,10\" horizon=500000 weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=1 hydra.sweep.dir=expablation/statlog_5010/run3 hydra.launcher.submitit_folder=expablation/statlog_5010/run3/.slurm

    python run_linearfinite.py -m domain=statlog.yaml algo=nnlinucb epsilon_decay=none check_glrt=false layers=\"50,50,50,50,10\" horizon=500000 weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=1,2 glrt_scale=1 hydra.sweep.dir=expablation/statlog_5010/run4 hydra.launcher.submitit_folder=expablation/statlog_5010/run4/.slurm

# covertype

    python run_linearfinite.py -m domain=covertype.yaml algo=nnegreedy epsilon_decay=cbrt,sqrt check_glrt=true layers=\"50,50,50,50,10\" horizon=1000000 weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=1,2,5 USE_MAXNORM=FALSE hydra.sweep.dir=expablation/covertype_5010/run1 hydra.launcher.submitit_folder=expablation/covertype_5010/run1/.slurm

    python run_linearfinite.py -m domain=covertype.yaml algo=nnlinucb epsilon_decay=none check_glrt=true layers=\"50,50,50,50,10\" horizon=1000000 weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=1,2 glrt_scale=1,2,5 USE_MAXNORM=FALSE hydra.sweep.dir=expablation/covertype_5010/run2 hydra.launcher.submitit_folder=expablation/covertype_5010/run2/.slurm

    python run_linearfinite.py -m domain=covertype.yaml algo=nnegreedy epsilon_decay=cbrt,sqrt check_glrt=false layers=\"50,50,50,50,10\" horizon=1000000  weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=1 hydra.sweep.dir=expablation/covertype_5010/run3 hydra.launcher.submitit_folder=expablation/covertype_5010/run3/.slurm

    python run_linearfinite.py -m domain=covertype.yaml algo=nnlinucb epsilon_decay=none check_glrt=false layers=\"50,50,50,50,10\" horizon=1000000  weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=1,2 glrt_scale=1 hydra.sweep.dir=expablation/covertype_5010/run4 hydra.launcher.submitit_folder=expablation/covertype_5010/run4/.slurm


--------------------------------------------------------------
# old

    python run_linearfinite.py -m domain=statlog.yaml algo=nnlinucb epsilon_decay=none check_glrt=true layers=\"50,50,50,50,10\" horizon=500000 weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=3 glrt_scale=5 hydra.sweep.dir=ablation/statlog_5010/run4 hydra.launcher.submitit_folder=ablation/statlog_5010/run4/.slurm

    python run_linearfinite.py -m domain=statlog.yaml algo=nnegreedy epsilon_decay=frt,cbrt,sqrt check_glrt=true layers=\"50,50,50,50,10\" horizon=500000 weight_rayleigh=1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=1,2 hydra.sweep.dir=ablation/statlog_5010/run5 hydra.launcher.submitit_folder=ablation/statlog_5010/run5/.slurm

    python run_linearfinite.py -m domain=statlog.yaml algo=nnlinucb epsilon_decay=none check_glrt=true layers=\"50,50,50,50,10\" horizon=500000 weight_rayleigh=1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=1 glrt_scale=1 hydra.sweep.dir=ablation/statlog_5010/run6 hydra.launcher.submitit_folder=ablation/statlog_5010/run6/.slurm

    python run_linearfinite.py -m domain=statlog.yaml algo=nnlinucb epsilon_decay=none check_glrt=true layers=\"50,50,50,50,10\" horizon=500000 weight_rayleigh=1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=2 glrt_scale=2 hydra.sweep.dir=ablation/statlog_5010/run7 hydra.launcher.submitit_folder=ablation/statlog_5010/run7/.slurm

    python run_linearfinite.py -m domain=statlog.yaml algo=nnlinucb epsilon_decay=none check_glrt=true layers=\"50,50,50,50,10\" horizon=500000 weight_rayleigh=1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=3 glrt_scale=5 hydra.sweep.dir=ablation/statlog_5010/run8 hydra.launcher.submitit_folder=ablation/statlog_5010/run8/.slurm

    python run_linearfinite.py -m domain=statlog.yaml algo=squarecb epsilon_decay=none check_glrt=false gamma_exponent=sqrt,cbrt gamma_scale=1,10,100 layers=\"50,50,50,50,10\" horizon=500000 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 hydra.sweep.dir=ablation/statlog_5010/run9 hydra.launcher.submitit_folder=ablation/statlog_5010/run9/.slurm

    python run_linearfinite.py -m domain=statlog.yaml algo=gradientucb epsilon_decay=none check_glrt=false bonus_scale=0.1,1,10 layers=\"50,50,50,50,10\" horizon=500000 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 hydra.sweep.dir=ablation/statlog_5010/run10 hydra.launcher.submitit_folder=ablation/statlog_5010/run10/.slurm

    python run_linearfinite.py -m domain=statlog.yaml algo=nnegreedy epsilon_decay=frt,cbrt,sqrt check_glrt=false layers=\"50,50,50,50,10\" horizon=500000 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 hydra.sweep.dir=ablation/statlog_5010/run11 hydra.launcher.submitit_folder=ablation/statlog_5010/run11/.slurm

    python run_linearfinite.py -m domain=statlog.yaml algo=nnlinucb epsilon_decay=none check_glrt=false layers=\"50,50,50,50,10\" horizon=500000 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=2 glrt_scale=2 hydra.sweep.dir=ablation/statlog_5010/run12 hydra.launcher.submitit_folder=ablation/statlog_5010/run12/.slurm