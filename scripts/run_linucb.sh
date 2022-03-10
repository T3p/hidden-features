#!/bin/bash

#CURDIR=`pwd`
#CODEDIR=`mktemp -d -p ${CURDIR}/tmp`
#cp ${CURDIR}/*.py ${CODEDIR}
#cp -r ${CURDIR}/a2c_ppo_acktr ${CODEDIR}

CDIR=/checkpoint/${USER}/linear-pruned-std-0.5
mkdir -p ${CDIR}
ALGO='linucb'

for BANDITTYPE in 'onehot' 'expanded'; do
for BONUS in 0.001 0.1; do
for SEED in 0 1 2 3 4 5 6 7 8 9; do
  SUBDIR=${ALGO}-bonus-${BONUS}-seed-${SEED}
  SAVEDIR=${CDIR}/${BANDITTYPE}/${SUBDIR}
  LOGDIR=${CDIR}/${BANDITTYPE}/logs/${SUBDIR}
  mkdir -p ${SAVEDIR}
  mkdir -p ${LOGDIR}
  JOBNAME=linear-bandit
  SCRIPT=${SAVEDIR}/run.sh
  SLURM=${SAVEDIR}/run.slurm
#  LOGDIR=${CDIR}/logs/${SUBDIR}
  extra=""
  echo "#!/bin/sh" > ${SCRIPT}
  echo "#!/bin/sh" > ${SLURM}
  echo "#SBATCH --job-name=${JOBNAME}" >> ${SLURM}
  echo "#SBATCH --output=${SAVEDIR}/stdout" >> ${SLURM}
  echo "#SBATCH --error=${SAVEDIR}/stderr" >> ${SLURM}
  echo "#SBATCH --partition=devlab" >> ${SLURM}
  echo "#SBATCH --nodes=1" >> ${SLURM}
  echo "#SBATCH --ntasks=1" >> ${SLURM}
  echo "#SBATCH --time=2:00:00" >> ${SLURM}
  echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
  echo "#SBATCH --signal=USR1" >> ${SLURM}
  echo "#SBATCH --gres=gpu:1" >> ${SLURM}
  echo "#SBATCH --mem=50G" >> ${SLURM}
  echo "#SBATCH -c 1" >> ${SLURM}
  echo "sh ${SCRIPT}" >> ${SLURM}
  echo "echo \$SLURM_JOB_ID >> ${SAVEDIR}/id" >> ${SCRIPT}
  echo "nvidia-smi" >> ${SCRIPT}
  echo /private/home/atouati/.conda/envs/xb.simple/bin/python /private/home/atouati/lrcb/run_linear_newtest.py \
    --bandittype ${BANDITTYPE} \
    --algo ${ALGO} \
    --seed ${SEED} \
    --bonus_scale ${BONUS} \
    --noise_std 0.5 \
    --device cpu \
    --horizon 10000 \
    --save_dir ${SAVEDIR} \
    --log_dir ${LOGDIR} >> ${SCRIPT}
  sbatch ${SLURM}
done
done
done