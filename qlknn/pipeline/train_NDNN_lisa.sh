#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=36:00:00

module load python/3.5.0-intel
module load mkl/64/11.0.5
export PATH=$PATH:$HOME/bin
module load netcdf/intel
source $HOME/tensorflow_env/qlknn/bin/activate

cd $PBS_O_WORKDIR

python -c 'from qlknn.training import train_NDNN; train_NDNN.train_NDNN_from_folder()'
