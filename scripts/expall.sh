#!/bin/bash

for DATASET in mushroom covertype magic statlog wheel
do
    HORIZON=500000
    if [[ $DATASET == *"covertype"* ]]; then
        HORIZON=1000000
    fi
    echo "#!/bin/bash" > run_${DATASET}.sh

    echo 'python run_linearfinite.py -m domain='${DATASET}'.yaml algo=nnegreedy epsilon_decay=cbrt,sqrt check_glrt=true layers=\"50,50,50,50,10\" horizon='${HORIZON}' weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=1,2,5,10,15 use_maxnorm=False device="cuda" hydra.sweep.dir=expablation/'${DATASET}'_5010/run1 hydra.launcher.submitit_folder=expablation/'${DATASET}'_5010/run1/.slurm &' >> run_${DATASET}.sh

    echo 'python run_linearfinite.py -m domain='${DATASET}'.yaml algo=nnlinucb epsilon_decay=none check_glrt=true layers=\"50,50,50,50,10\" horizon='${HORIZON}' weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=1,2 glrt_scale=1,2,5,10,15 use_maxnorm=False device="cuda" hydra.sweep.dir=expablation/'${DATASET}'_5010/run2 hydra.launcher.submitit_folder=expablation/'${DATASET}'_5010/run2/.slurm & ' >> run_${DATASET}.sh

    echo 'python run_linearfinite.py -m domain='${DATASET}'.yaml algo=nnegreedy epsilon_decay=cbrt,sqrt check_glrt=false layers=\"50,50,50,50,10\" horizon='${HORIZON}' weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=1 use_maxnorm=False device="cuda" hydra.sweep.dir=expablation/'${DATASET}'_5010/run3 hydra.launcher.submitit_folder=expablation/'${DATASET}'_5010/run3/.slurm & ' >> run_${DATASET}.sh

    echo 'python run_linearfinite.py -m domain='${DATASET}'.yaml algo=nnlinucb epsilon_decay=none check_glrt=false layers=\"50,50,50,50,10\" horizon='${HORIZON}' weight_min_features=0,1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=1,2 glrt_scale=1 use_maxnorm=False device="cuda" hydra.sweep.dir=expablation/'${DATASET}'_5010/run4 hydra.launcher.submitit_folder=expablation/'${DATASET}'_5010/run4/.slurm & ' >> run_${DATASET}.sh

    echo 'python run_linearfinite.py -m domain='${DATASET}'.yaml algo=nnegreedy epsilon_decay=cbrt,sqrt check_glrt=true layers=\"50,50,50,50,10\" horizon='${HORIZON}' weight_rayleigh=1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 glrt_scale=1,2,5,10,15 use_maxnorm=False device="cuda" hydra.sweep.dir=expablation/'${DATASET}'_5010/run5 hydra.launcher.submitit_folder=expablation/'${DATASET}'_5010/run5/.slurm &' >> run_${DATASET}.sh

    echo 'python run_linearfinite.py -m domain='${DATASET}'.yaml algo=nnlinucb epsilon_decay=none check_glrt=true layers=\"50,50,50,50,10\" horizon='${HORIZON}' weight_rayleigh=1 seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 bonus_scale=1,2 glrt_scale=1,2,5,10,15 use_maxnorm=False device="cuda" hydra.sweep.dir=expablation/'${DATASET}'_5010/run2 hydra.launcher.submitit_folder=expablation/'${DATASET}'_5010/run2/.slurm & ' >> run_${DATASET}.sh

    echo 'python run_linearfinite.py -m domain='${DATASET}'.yaml algo=squarecb epsilon_decay=none check_glrt=false gamma_exponent=sqrt,cbrt gamma_scale=1,10,50,100 layers=\"50,50,50,50,10\" horizon='${HORIZON}' seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 device="cuda" hydra.sweep.dir=expablation/'${DATASET}'_5010/run9 hydra.launcher.submitit_folder=expablation/'${DATASET}'_5010/run9/.slurm & ' >> run_${DATASET}.sh

    echo 'python run_linearfinite.py -m domain='${DATASET}'.yaml algo=gradientucb epsilon_decay=none check_glrt=false bonus_scale=0.1,1,2,5 layers=\"50,50,50,50,10\" horizon='${HORIZON}' seed=1,48121,6598,90144,44310,88361,43482,74508,32279,93111,30145,80831,88824,54953,5967,11579,46670,31024,56024,62782 device="cuda" hydra.sweep.dir=expablation/'${DATASET}'_5010/run10 hydra.launcher.submitit_folder=expablation/'${DATASET}'_5010/run10/.slurm & ' >> run_${DATASET}.sh

    chmod +x run_${DATASET}.sh
    echo "written run_${DATASET}.sh"
done