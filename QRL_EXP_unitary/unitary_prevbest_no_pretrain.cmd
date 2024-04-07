#!/bin/bash
#PBS -e errorfile.err
#PBS -o logfile.log
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q gpuq
echo $PBS_O_WORKDIR
tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job/$tpdir
mkdir -p $tempdir
echo $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .
module load anaconda3_2023
module load cuda12.1
conda_executable=$(which conda)
echo $conda_executable
source /lfs/sware/anaconda3_2023/etc/profile.d/conda.sh

conda env list
conda activate qrl
which python3
conda list
pip3 install qiskit==0.43.0 --no-cache-dir
pip3 install qiskit-machine-learning==0.6.1 --no-cache-dir
pip3 install gym==0.26 --no-cache-dir
pip3 install pylatexenc --no-cache-dir
pip3 install scikit-learn==1.1.1 numpy==1.22.1 pygments==2.15.1 matplotlib==3.7.1 pandas Pillow wandb --no-cache-dir
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --no-cache-dir
wandb sync --clean
python3 -u "prevbest_no_pretrain.py"
conda deactivate
cp -R * $PBS_O_WORKDIR/.
rm -rf $tempdir/*
cd ~
source .bash_profile