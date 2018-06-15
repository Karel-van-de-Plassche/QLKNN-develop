#!/bin/bash -l

##SBATCH --qos=regular
#SBATCH --qos=premium
#SBATCH --nodes=1
#SBATCH --time 24:00:00
#SBATCH -L SCRATCH
#SBATCH -C haswell

module load cray-hdf5
module load cray-netcdf
module load tensorflow/intel-head-python3

cd $SLURM_SUBMIT_DIR

python -c 'from qlknn.training import train_NDNN; train_NDNN.train_NDNN_from_folder()'
