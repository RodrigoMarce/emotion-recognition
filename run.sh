#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=64g
#SBATCH -J "Face analysis"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:2
#SBATCH -C A100|V100

source ~/miniconda3/etc/profile.d/conda.sh
conda activate face
python emotion-recognition/emotion_recognition_test_gray.py -i emotion-recognition/test.mov -o out.mp4
