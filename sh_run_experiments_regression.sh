#!/bin/sh

P_dataset="CaliH"
Q_dataset="KingH"
N=10
size=2000


python regression/Ours_conditional.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset} -kde
python regression/ACC.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset} 
python regression/Ours.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset} -gmm -nocuda
# python regression/Ours.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset} -kde -nocuda


P_dataset="Census15"
Q_dataset="Census17"
N=5
size=2000



python regression/Ours_conditional.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset} -kde
python regression/ACC.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset} 
python regression/Ours.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset} -gmm -nocuda
# python regression/Ours.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset} -kde -nocuda


P_dataset="Census15"
Q_dataset="Census17"
N=5
size=4000

python regression/Ours_conditional.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset} -kde
python regression/ACC.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset} 
python regression/Ours.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset} -gmm -nocuda
# python regression/Ours.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset} -kde -nocuda


P_dataset="Census15"
Q_dataset="Census17"
N=10
size=2000

python regression/Ours_conditional.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset} -kde
python regression/ACC.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset} 
python regression/Ours.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset} -gmm -nocuda
# python regression/Ours.py -N ${N} --size ${size} -P ${P_dataset} -Q ${Q_dataset} -kde -nocuda

