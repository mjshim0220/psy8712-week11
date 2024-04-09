#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBSTCH --mem=16gb
#SBATCH -t 01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shim0220@umn.edu
#SBATCH -p amdsmall
cd ~/psy8712-week11/msi
module load R/4.3.0-openblas
Rscript week11-cluster.R