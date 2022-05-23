HORIZON=400000
SEEDS=3213,2135,21,5642,9621

########
# LINUCB
########

# MSE
python run_linearfinite.py -m domain=statlog.yaml horizon=${HORIZON} algo=nnlinucb check_glrt=True glrt_scale=1,2,4 bonus_scale=0.5,1,2 layers=\"100,100,50,20,10\",\"100,100\",\"100\" weight_mse=1 weight_rayleigh=0 weight_min_features=0 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=${SEEDS} use_tb=True use_wandb=False use_maxnorm=True epsilon_decay=none device="cpu" hydra.sweep.dir=statlog_0523/\${algo}_mse hydra.launcher.submitit_folder=statlog_0523/\${algo}_mse/.slurm save_model_at_train=True

# min feat
python run_linearfinite.py -m domain=statlog.yaml horizon=${HORIZON} algo=nnlinucb check_glrt=True glrt_scale=1,2,4 bonus_scale=0.5,1,2 layers=\"100,100,50,20,10\",\"100,100\",\"100\" weight_mse=1 weight_rayleigh=0 weight_min_features=0.5,1,2 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=${SEEDS} use_tb=True use_wandb=False use_maxnorm=False epsilon_decay=none device="cpu" hydra.sweep.dir=statlog_0523/\${algo}_minfeat hydra.launcher.submitit_folder=statlog_0523/\${algo}_minfeat/.slurm save_model_at_train=True

# rayleigh
python run_linearfinite.py -m domain=statlog.yaml horizon=${HORIZON} algo=nnlinucb check_glrt=True glrt_scale=1,2,4 bonus_scale=0.5,1,2 layers=\"100,100,50,20,10\",\"100,100\",\"100\" weight_mse=1 weight_rayleigh=0.5,1,2 weight_min_features=0 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=${SEEDS} use_tb=True use_wandb=False use_maxnorm=False epsilon_decay=none device="cpu" hydra.sweep.dir=statlog_0523/\${algo}_rayleigh hydra.launcher.submitit_folder=statlog_0523/\${algo}_rayleigh/.slurm save_model_at_train=True


########
# E-GREEDY
########

# MSE
python run_linearfinite.py -m domain=statlog.yaml horizon=${HORIZON} algo=nnegreedy check_glrt=True glrt_scale=1,2,4 bonus_scale=1 layers=\"100,100,50,20,10\",\"100,100\",\"100\" weight_mse=1 weight_rayleigh=0 weight_min_features=0 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=${SEEDS} use_tb=True use_wandb=False use_maxnorm=True epsilon_decay=cbrt,sqrt device="cpu" hydra.sweep.dir=statlog_0523/\${algo}_mse hydra.launcher.submitit_folder=statlog_0523/\${algo}_mse/.slurm save_model_at_train=True

# min feat
python run_linearfinite.py -m domain=statlog.yaml horizon=${HORIZON} algo=nnegreedy check_glrt=True glrt_scale=1,2,4 bonus_scale=1 layers=\"100,100,50,20,10\",\"100,100\",\"100\" weight_mse=1 weight_rayleigh=0 weight_min_features=0.5,1,2 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=${SEEDS} use_tb=True use_wandb=False use_maxnorm=False epsilon_decay=cbrt,sqrt device="cpu" hydra.sweep.dir=statlog_0523/\${algo}_minfeat hydra.launcher.submitit_folder=statlog_0523/\${algo}_minfeat/.slurm  save_model_at_train=True

# rayleigh
python run_linearfinite.py -m domain=statlog.yaml horizon=${HORIZON} algo=nnegreedy check_glrt=True glrt_scale=1,2,4 bonus_scale=1 layers=\"100,100,50,20,10\",\"100,100\",\"100\" weight_mse=1 weight_rayleigh=0.5,1,2 weight_min_features=0 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=${SEEDS} use_tb=True use_wandb=False use_maxnorm=False epsilon_decay=cbrt,sqrt device="cpu" hydra.sweep.dir=statlog_0523/\${algo}_rayleigh hydra.launcher.submitit_folder=statlog_0523/\${algo}_rayleigh/.slurm  save_model_at_train=True
