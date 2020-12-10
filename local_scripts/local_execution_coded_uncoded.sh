#!/usr/bin/ksh

# 4 agents
#python3 ../experiments/train.py --num_agents 4 --scenario simple_spread


#mpirun -n 6 python3 ../experiments/maddpg_uncoded.py --scenario simple_spread --num_agents 4


# 5 agents
#python3 ../experiments/train.py --num_agents 5 --scenario simple_spread


#mpirun -n 7 python3 ../experiments/maddpg_uncoded.py --scenario simple_spread --num_agents 5


# 8 agents
python3 ../experiments/train.py --num_agents 8 --scenario simple_spread


#mpirun -n 10 python3 ../experiments/maddpg_uncoded.py --scenario simple_spread --num_agents 8
