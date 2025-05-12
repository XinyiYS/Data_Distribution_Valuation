#!/bin/sh

P_dataset="CIFAR10"
Q_dataset="CIFAR100"
N=5
size=10000

python run_Ours_conditional.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset}  


P_dataset="CIFAR10"
Q_dataset="CIFAR100"
N=10
size=10000

python run_Ours_conditional.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset}  

# P_dataset="MNIST"
# Q_dataset="EMNIST"
# N=5
# size=10000

# python run_Ours_conditional.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset}  


# P_dataset="MNIST"
# Q_dataset="FaMNIST"
# N=5
# size=10000

# python run_Ours_conditional.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset}  


# P_dataset="MNIST"
# Q_dataset="EMNIST"
# N=10
# size=10000

# python run_Ours_conditional.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset}  


# P_dataset="MNIST"
# Q_dataset="FaMNIST"
# N=10
# size=10000

# python run_Ours_conditional.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset}  



P_dataset="TON"
Q_dataset="UGR16"
N=5
size=4000

python run_Ours_conditional.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset}  



P_dataset="CreditCard"
Q_dataset="CreditCard"
N=5
size=10000
python run_Ours_conditional.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset}  


P_dataset="CreditCard"
Q_dataset="CreditCard"
N=5
size=5000

python run_Ours_conditional.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset}  
