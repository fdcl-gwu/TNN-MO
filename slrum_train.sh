#!/bin/bash

#SBATCH --job-name=MuWiH
#SBATCH -o logs/JobID_%j_nodeID_%n.out
#SBATCH -e logs/JobID_%j_nodeID_%n.err 
#SBATCH --nodes=1
# SBATCH --ntasks=4
# SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=10000
#SBATCH -p large-gpu
#SBATCH -t 7-00:00:00 

CONFIG_DIR=""

Resume="ON"

if [ "$Resume" = "OFF" ]; then
    CONFIG_DIR=""
fi

if [ "$Resume" = "ON" ]; then
    CKPT="/home/maneesh/Desktop/LAB2.0/DATA-FDCL/Checkpoints/"
    CONFIG_DIR="E01rev0_10-15-2023_Multi-2X(DGH.WH)__DETR-1.0-NQ2_NC1__DT50"
fi


if [ "$Resume" = "OFF" ]; then
    CONFIG=$CONFIG_DIR"config.ini"
    echo "\033[36mConverting pascal format ship data to coco format\033[0m"
    python3 scripts/ship_to_coco.py --configfile $CONFIG
    echo "\033[36mSTART training .......\033[0m"
    python3 scripts/main.py --configfile $CONFIG
fi

if [ "$Resume" = "ON" ]; then
    echo "\033[36mSTART training .......\033[0m"
    CONFIG=$CKPT$CONFIG_DIR"/config.ini"
    python3 scripts/main.py --configfile $CONFIG
fi


