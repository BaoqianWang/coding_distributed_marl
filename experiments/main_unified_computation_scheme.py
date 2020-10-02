from mpi4py import MPI
import mpi4py
from central_controller import CentralController
import tensorflow as tf
from learner import Learner
import time
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import time
from statistics import mean
from RandomLinear_code import *
from ReplicationBased import *
from MDS_code import *
from RegularLDPC_code import *
import argparse
from numpy.linalg import matrix_rank
import json


# Set the random number generation seed
random.seed(30)

def parse_args():
    parser = argparse.ArgumentParser("Distributed Multi-agent RL")
    # Environment
    parser.add_argument("--scheme", type=str, default="hcmm", help="computation schemes including replication-based, mds, ...")
    parser.add_argument("--num_straggler", type=int, default="0", help="num straggler")

    return parser.parse_args()



def make_env(num_agents, num_landmarks,scenario_name="simple_spread_comm"):
    from multiagent.environment_comm import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(num_agents, num_landmarks)
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    return env

def get_learners(env, session, name, num_learners, obs_shape_n, num_actions):
    Learners=[]
    for i in range(num_learners):
        Learners.append(Learner(env,session, name+"agent_%d" %i,obs_shape_n,num_actions,i,gamma,epsilon))

    return Learners

def interact_with_environment(env, Learners, steps, display = False):
    step=0
    obs=env.reset()
    # print('The observation is ',obs)
    # observations=[]
    # observations.append(obs)
    rewards=[]
    # trajectory=[]
    while step<steps:

        actions=[learner.action(obs) for learner in Learners]

        next_obs, rew_n, done_n, info_n = env.step(actions)
        rewards.append(sum(rew_n))

        if (display == True):
            time.sleep(1)
            env.render()
        #Each agent add experience
        for i, learner in enumerate(Learners):
            learner.observation.append(obs[0])
            learner.actions.append(actions[i])
            learner.reward.append(rew_n[i])

        obs=copy.deepcopy(next_obs)

        step+=1
    #print('')
    return mean(rewards)


if __name__=="__main__":

    arglist = parse_args()
    #with open('marl_parameters.json') as f:
    with open('/home/ubuntu/marl_coding/marl_parameters.json') as f:
        parameters = json.load(f)

    gamma = parameters['gamma']
    epsilon = parameters['epsilon']
    num_agents = parameters['num_agents']
    num_learners = parameters['coded_num_learners']
    num_actions = parameters['num_actions']
    num_straggler = arglist.num_straggler
    reward_episode = parameters['reward_episode']
    time_steps = parameters['time_steps']
    max_iteration = parameters['max_iteration']


    with tf.Session() as session:
        num_train=0

        tf.set_random_seed(30)
        random.seed(30)
        # MPI initialization.
        comm = MPI.COMM_WORLD
        num_node = comm.Get_size()
        assert num_node == num_learners +2
        node_id = comm.Get_rank()
        node_name = MPI.Get_processor_name()
        status = MPI.Status()

        CENTRAL_CONTROLLER = [0]
        ACTOR = [1]
        LEARNERS = [i for i in range(2, 2+num_learners)]


        # Determine the computation scheme
        if(arglist.scheme == "RandomLinear"):
            code_scheme = RandomLinear_code(num_agents, num_learners, 0.7)
        if(arglist.scheme == "MDS"):
            code_scheme =  MDS_code(num_agents, num_learners) #Need to modify the mds code class
        if(arglist.scheme == "ReplicationBased"):
            code_scheme = ReplicationBased(num_agents, num_learners)
        if(arglist.scheme == "RegularLDPC"):
            #code_scheme = RegularLDPC_code(8, 3, 4)
            code_scheme = RegularLDPC_code(10, 2, 10)

        H = code_scheme.H


        # Create environment and learner network for all controllers, actor and learners
        env = make_env(num_agents, num_agents)
        obs_shape_n = [env.observation_space[0].shape[0]]

        Learners=get_learners(env, session, "node_id_%d" %node_id, num_agents, obs_shape_n, num_actions)

        session.run(tf.global_variables_initializer())

        weight_length = len(Learners[0].get_local_weigths())
        decoding_time = 0
        weight_communication_time = 0
        env_communication_time = 0

        episode_reward = []
        run_time=[]
        plot_reward=[]

        start_time=time.time()
        done=0



        while True:
            STRAGGLER = random.sample(LEARNERS, num_straggler)
            comm.Barrier()

            if(node_id in CENTRAL_CONTROLLER):
                iteration_start_time = time.time()

                weight_start_time = time.time()
                agent_weights = [Learners[i].get_local_weigths() for i in range(num_agents)]
                for i, row in enumerate(H):
                    all_agents_weights = [None] * num_agents
                    for j, entry in enumerate(row):
                        if entry != 0:
                            all_agents_weights[j] = copy.deepcopy(agent_weights[j])
                    comm.send(all_agents_weights, dest=2+i, tag=num_train)

                comm.send(agent_weights, dest=1, tag=num_train)
                weight_end_time = time.time()
                weight_communication_time += weight_end_time - weight_start_time


                received_data = []
                received_matrix = []
                #weight=None

                #for i in range(num_agents):
                while True:
                     req=comm.irecv(source=MPI.ANY_SOURCE, tag=num_train)
                     data=req.wait()
                     received_data.append(data)
                     received_matrix.append(H[data[0]])
                     #print(received_matrix)
                     if(matrix_rank(np.asarray(received_matrix))>=num_agents):
                         break

                start_encoding=time.time()
                decoded_weights = code_scheme.decode(received_data, weight_length)
                end_encoding=time.time()
                decoding_time += end_encoding - start_encoding
                for i in range(num_agents):
                    Learners[i].set_weights(decoded_weights[i])


                #ack_start_time=time.time()
                for id  in LEARNERS:
                    comm.send(1, dest=id, tag=num_train)


                num_train+=1
                if num_train >= max_iteration:
                    #print('Decoding time is', decoding_time)
                    print('Weight Communication Time is', weight_communication_time)
                    break

            else:
                all_agents_weights = comm.recv(source=0, tag=num_train)
                for i, agent_weight in enumerate(all_agents_weights):
                    if agent_weight is not None:
                        Learners[i].set_weights(agent_weight)



            if(node_id in ACTOR):
                interaction_start_time = time.time()
                reward_average = interact_with_environment(env, Learners, time_steps)
                interaction_end_time = time.time()

                episode_reward.append(reward_average)

                episodes_total = []

                #Send environment data according to encoding matrix
                env_start_time = time.time()
                for i, row in enumerate(H):
                    episodes_total=[None]*num_agents

                    for j, entry in enumerate(row):
                        if entry != 0:
                            episodes = [Learners[j].observation, Learners[j].actions, Learners[j].reward]
                            episodes_total[j] = episodes

                    comm.send(episodes_total, dest=2+i, tag=num_train)
                env_end_time = time.time()

                env_communication_time += env_end_time-env_start_time

                if(num_train%reward_episode==0 and num_train >0):
                    end_time=time.time()
                    print("steps: {}, episode reward: {}, time: {}".format(num_train, np.mean(episode_reward[-reward_episode:]), end_time-start_time))
                    run_time.append(end_time - start_time)
                    start_time=time.time()
                    plot_reward.append(np.mean(episode_reward[-reward_episode:]))

                for i in range(num_agents):
                    Learners[i].reset_episode()

                num_train+=1
                #print('Interaction time', interaction_end_time-interaction_start_time)
                if num_train >= max_iteration:
                    print("Average Execution Time for %d" %reward_episode, np.mean(run_time[1:]))
                    print('env_communication_time', env_communication_time)
                    #plt.figure(1)
                    #plt.plot(plot_reward)
                    #plt.show()
                    #plt.xlabel('Number of iterations')
                    #plt.ylabel('Globally Average Reward')
                    #plt.show()
                    break



            #Learners start training, updating, and sending weights back to the central controller
            if(node_id in LEARNERS):

                episode_start = time.time()
                episodes = comm.recv(source=1, tag=num_train)
                episode_end = time.time()
                weights = [None] * num_agents


                start_learn_time = time.time()
                for i, entry in enumerate(H[node_id-2]):
                    if (entry != 0):
                        Learners[i].observation = episodes[i][0]
                        Learners[i].actions = episodes[i][1]
                        Learners[i].reward = episodes[i][2]
                        Learners[i].learn()
                        weights[i] = Learners[i].get_local_weigths()

                sum_weights=[]

                for k in range(weight_length):
                    weight = 0
                    for i, entry in enumerate(H[node_id-2]):
                        if (entry != 0):
                            weight += entry*weights[i][k]
                    sum_weights.append(weight)
                end_learn_time = time.time()
                data = [node_id-2, sum_weights]


                if(node_id in STRAGGLER):

                    clock = 0
                    req=comm.irecv(source=0,tag=num_train)
                    while True:
                        if(req.Test()):
                            break
                        if(clock >=.2):
                            comm.isend(data, dest=0,tag=num_train)
                            req.Wait()
                            break
                        time.sleep(0.001)
                        clock+=0.001

                else:
                    comm.isend(data, dest=0, tag=num_train)
                    comm.recv(source=0, tag=num_train)


                end_worker_weights=time.time()

                num_train+=1

                if num_train >= max_iteration:
                    break
