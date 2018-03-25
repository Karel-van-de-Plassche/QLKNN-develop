#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=FUA32_WPJET1
#SBATCH --partition=skl_fua_prod
#SBATCH --time=24:00:00


export PATH=$PATH:$HOME/bin
module load profile/global
module load intel/pe-xe-2017--binary #Needed for numpy
module load mkl/2017--binary #Needed for numpy
module load szip/2.1--gnu--6.1.0 #Needed for hdf5
module load zlib/1.2.8--gnu--6.1.0 #Needed for hdf5
module load intelmpi/2017--binary #Needed for hdf5
module load hdf5/1.8.17--intelmpi--2017--binary
export HDF5_DIR=$HDF5_HOME
module load netcdf/4.4.1--intel--pe-xe-2017--binary
export NETCDF4_DIR=$NETCDF_HOME

module load gnu/6.1.0
module load python/2.7.12
module load numpy/1.13.0--python--2.7.12
module load tensorflow/1.5.0--python--2.7.12
module load scipy/0.19.0--python--2.7.12

cd $SLURM_SUBMIT_DIR

python -c 'from qlknn.training import train_NDNN; train_NDNN.train_NDNN_from_folder()'
