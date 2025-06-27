#!/bin/bash

#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

module load python/3.11.5
module load scipy-stack
module load rdkit

virtualenv --no-download $SLURM_TMPDIR GaUDI_env
source $SLURM_TMPDIR/GaUDI_env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r /home/agiguere/projects/def-agiguere/Data/requirement/requirements_gaudi.txt
pip install --no-index cairocffi cairosvg

mkdir DATASETS
python /home/agiguere/projects/def-agiguere/GaUDI/generator-main_v2/gen_contacts.py -d /home/agiguere/projects/def-agiguere/GaUDI/compas-3x_planar05.csv
mv DATASETS/MODIFIED_COMPAS-1D.csv DATASETS/MODIFIED_COMPAS-3x.csv

deactivate
