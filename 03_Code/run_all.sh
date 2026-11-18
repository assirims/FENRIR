#!/bin/bash
# Runs every federated experiment group on worker $1 of $2 (resumable).
W=$1; N=$2
for g in main robust noniid central ablation; do
  OMP_NUM_THREADS=1 python3 exp_fl.py $g $W $N
done
