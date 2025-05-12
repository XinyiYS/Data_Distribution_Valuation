#!/bin/sh

# to rerun all the experiments

P_dataset="MNIST"
Q_dataset="FaMNIST"
N=5
size=10000

python run_Ours.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset}  
python run_ACC.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset}  

