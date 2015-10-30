#!/bin/bash
./split -p 6 -e 0.001 train.txt
./sendtotrain.py machinefile train.txt temp_dir
mpirun -n 6 --machinefile machinefile ./train -e 1e-5 /home/jing/dis_data/train.txt.sub
./splittopredict.py machinefile test.txt
mpirun -n 6 --machinefile machinefile ./predict /home/jing/dis_data/test.txt.sub train.txt.sub.model MQ2007.txt
