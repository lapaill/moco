#!/bin/bash
#SBATCH --job-name=8053_mocoresnet18 
#SBATCH -p gpu-cbio
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task 12
#SBATCH --mem 100000
#SBATCH --output=slurm_out/Hflip-Vflip-GrayScale-GaussianBlur-Rotate90-Jitter-MultipleElasticDistort-200.out  # Nom du fichier de sortie contenant l'ID et l'indice 
#SBATCH --error=slurm_out/Hflip-Vflip-GrayScale-GaussianBlur-Rotate90-Jitter-MultipleElasticDistort-200.err   # Nom du fichier d'erreur (ici commun avec la sortie) 

module load cuda10.0
python /mnt/data4/jlaval/moco/main_moco_histo.py \
    -a resnet18 \
    --mlp \
    --moco-t 0.2 \
    --cos \
    --lr 0.003 \
    --batch-size 4096 \
    --data /mnt/data4/tlazard/data/dataset_stage_2/data \
    --rank 0 \
    --world-size 1 \
    --multiprocessing-distributed \
    --dist-url "tcp://127.0.0.1:8053" \
    --name_expe Hflip-Vflip-GrayScale-GaussianBlur-Rotate90-Jitter-MultipleElasticDistort-200 \
    --epochs 200 \
    --crop-and-transform \
    --transformations Hflip Vflip GrayScale GaussianBlur Rotate90 Jitter MultipleElasticDistort
#    transformations [ Hflip, Vflip, Crop, Crop_and_rotate, HEaug, GrayScale, MultipleElasticDistort, GaussianBlur, Jitter, Rotate90 ]
#   /!\ Ne pas mettre d'espace après les '\' de retour à la ligne 
