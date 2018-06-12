#!/bin/bash
#MSUB -l nodes=1:ppn=10
#MSUB -l walltime=00:15:00
#MSUB -l pmem=6400mb
#MSUB -N tensorflow_minesweeper
#PBS -o log.txt

module load devel/python/3.5.2
source ./venv/bin/activate
echo "Loaded virtual env"
cd rl_agent
echo "Running on the following python version"
python --version
echo "Start training"
python train.py
echo "Finished training"
deactivate
