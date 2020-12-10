import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI


import random
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg_coding_replay_memory import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import json
from coding_framework.MDS_code import MDS_code
from coding_framework.ReplicationBased import ReplicationBased
from coding_framework.RandomLinear_code import RandomLinear_code
from coding_framework.VandermondeLDPC_code import VandermondeLDPC_code
import copy
from numpy.linalg import matrix_rank
from experiments.common_configuration import parse_args, mlp_model, make_env, get_trainers


def interact_with_environments(env, trainers, steps, first_time = True):

    obs_n = env.reset()
    if (first_time):
        episode_rewards = [0.0]
        step = 0
        while True:
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            step += 1
            done = all(done_n)
            terminal = (step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i])

            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew

            if done or terminal:
                obs_n = env.reset()
                step = 0

            if len(trainers[0].replay_buffer) >= arglist.max_episode_len * arglist.batch_size:
                break
    else:
        episode_rewards = [0.0]
        step = 0
        for _ in range(steps):
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            step += 1
            done = all(done_n)
            terminal = (step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i])

            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew

            if done or terminal:
                episode_rewards.append(0)
                obs_n = env.reset()
                step = 0

    return episode_rewards

def get_trainers(env, num_agents, name, obs_shape_n, arglist, session):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_agents):
        trainers.append(trainer(
            name+"agent_%d" % i, model, obs_shape_n, session,env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))

    return trainers


if __name__=="__main__":


    arglist = parse_args()

    gamma = arglist.gamma
    steps = arglist.max_episode_len
    num_straggler = arglist.num_straggler
    num_learners = arglist.num_learners

    with tf.Session() as session:
        num_train=0

        tf.set_random_seed(30)
        random.seed(30)

        # MPI initialization.
        comm = MPI.COMM_WORLD
        num_node = comm.Get_size()

        node_id = comm.Get_rank()
        node_name = MPI.Get_processor_name()


        CENTRAL_CONTROLLER = [0]
        ACTOR = [1]


        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_agents = env.n

        LEARNERS = [i for i in range(2, 2+num_learners)]

        assert num_node == num_learners +2
        assert num_learners > num_agents

        # Determine the computation scheme
        if(arglist.scheme == "RandomLinear"):
            code_scheme = RandomLinear_code(num_agents, num_learners, 0.8)
        if(arglist.scheme == "MDS"):
            code_scheme =  MDS_code(num_agents, num_learners) #Need to modify the mds code class
        if(arglist.scheme == "ReplicationBased"):
            code_scheme = ReplicationBased(num_agents, num_learners)
        if(arglist.scheme == "RegularLDPC"):
            code_scheme = RegularLDPC_code(10, 2, 10)
        if(arglist.scheme) == "VandermondeLDPC":
            code_scheme = VandermondeLDPC_code(arglist.vanLDPC_p, arglist.vanLDPC_pho , arglist.vanLDPC_gamma)


        H = code_scheme.H
        #print(H)
        #print(H)

        weight_key = ['p_variables','target_p_variables','q_variables','target_q_variables']
        trainers = get_trainers(env, num_agents, "actor", obs_shape_n, arglist, session)
        U.initialize()

        weight_length = len(trainers[0].get_weigths()['p_variables'])
        weight_shape = []
        max_weight_size = 0
        max_weight_shape = None
        for i in range(num_agents):

            single_agent_weight = dict()

            for key in weight_key:
                temp = []
                for j in range(weight_length):
                    #temp_max_size =copy.deepcopy(max_weight_size)
                    #temp_max_shape = copy.deepcopy(max_weight_shape)
                    temp_shape = trainers[i].get_weigths()[key][j].shape
                    temp_size = trainers[i].get_weigths()[key][j].size
                    temp.append(temp_shape)
                    if(temp_size > max_weight_size):
                        max_weight_size = copy.deepcopy(temp_size)
                        max_weight_shape = copy.deepcopy(temp_shape)

                single_agent_weight[key] = copy.deepcopy(temp)

            weight_shape.append(copy.deepcopy(single_agent_weight))


        #weight_key = ['p_variables', 'target_p_variables', 'q_variables', 'target_q_variables']
        episode_rewards = []
        episodes = []
        final_episode_rewards = []
        train_time = []
        train_step = 0
        iter_step = 0

        if (node_id in ACTOR):
            print('Coded computation scheme: ', arglist.scheme)
            print('Scenario: ', arglist.scenario)
            print('Number of learners: ', num_learners)
            print('Number of agents: ', num_agents)
            if(arglist.scheme) == "VandermondeLDPC":
                print('LDPC parameters: ', arglist.vanLDPC_p, arglist.vanLDPC_pho , arglist.vanLDPC_gamma)
            interact_with_environments(env, trainers, steps, True)


        start=time.time()
        done = 0


        while True:
            STRAGGLER = random.sample(LEARNERS, num_straggler)


            # Central controller broadcast weights
            if(node_id in CENTRAL_CONTROLLER):
                all_agents_weights=[]
                for i, agent in enumerate(trainers):
                    all_agents_weights.append(agent.get_weigths())
            else:
                all_agents_weights = None

            ### Broadcast weights to all learners
            start_master_weights=time.time()
            all_agents_weights=comm.bcast(all_agents_weights,root=0)
            end_master_weights=time.time()

            #print('Broadcast done')

            if(node_id not in CENTRAL_CONTROLLER):
                for i, agent in enumerate(trainers):
                    agent.set_weigths(all_agents_weights[i])


            if(node_id in ACTOR):
                rewards = interact_with_environments(env, trainers, 100, False)
                rewards.pop()
                #print('rewards is', rewards)
                episode_rewards += rewards


                if(len(episode_rewards)%arglist.save_rate==0):
                    episode_time=time.time()
                    train_time.append(episode_time-start)
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(num_train, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),episode_time-start))
                    final_episode_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                    start=time.time()

                #print('Actor collecting number of %d episode' %(len(episode_rewards)), 'with train step', train_step)
                #if(train_step%100==0 and train_step > arglist.batch_size * arglist.max_episode_len):
                #episodes_total = []
                #for l in range(num_agents):
                replay_obs_n = []
                replay_obs_next_n = []
                replay_act_n = []
                replay_reward_n = []
                replay_done_n = []
                episodes = dict()

                index = trainers[0].replay_buffer.make_index(arglist.batch_size)
                for i in range(num_agents):
                    obs, act, rew, obs_next, done = trainers[i].replay_buffer.sample_index(index)
                    replay_obs_n.append(obs)
                    replay_obs_next_n.append(obs_next)
                    replay_act_n.append(act)
                    replay_reward_n.append(rew)
                    replay_done_n.append(done)

                episodes['observation']=replay_obs_n
                episodes['actions']=replay_act_n
                episodes['observation_next']=replay_obs_next_n
                episodes['reward']=replay_reward_n
                episodes['done']=replay_done_n
                #episodes_total.append(episodes)

                # for l, row in enumerate(H):
                #     episodes_total_send = [None]*num_agents
                #     for j, entry in enumerate(row):
                #         if entry != 0:
                #             episodes_total_send[j] = episodes_total[j]

            episodes = comm.bcast(episodes, root=1)


            if(node_id in ACTOR):
                #print(num_train)
                num_train += 1
                #print(num_train)
                if num_train > arglist.num_train:
                    print('The mean time is', np.mean(train_time[1:]), 'The corresponding variance is', np.var(train_time[1:]))
                    break


            if(node_id in LEARNERS):
                loss = None
                #episodes_total = comm.recv(source=1, tag=num_train)
                weights = [None] * num_agents

                for i, entry in enumerate(H[node_id-2]):
                    if (entry != 0):
                        #episodes = episodes_total[i]
                        loss = trainers[i].update(trainers, episodes)
                        weights[i] = trainers[i].get_weigths()

                sum_weights_dict = dict()

                for key in weight_key:
                    sum_weights = []
                    for k in range(weight_length):

                        weight = 0
                        for j, entry in enumerate(H[node_id -2]):
                            if (entry != 0):
                                weights[j][key][k].resize(max_weight_shape)
                                weight += entry*weights[j][key][k]
                        sum_weights.append(weight)

                    sum_weights_dict[key] = copy.deepcopy(sum_weights)

                data = [node_id-2, sum_weights_dict]
                #print('Start', num_train)
                #print(STRAGGLER)
                if(node_id in STRAGGLER):
                    clock = 0
                    req = comm.irecv(source=0, tag=num_train)
                    while True:
                        if(req.Test()):
                            break
                        if(clock >= 1):
                            req_send = comm.isend(data, dest=0, tag=num_train)
                            req.wait()
                            req_send.Cancel()
                            break
                        time.sleep(0.001)
                        clock+=0.001
                else:
                    #start_worker_weights=time.time()
                    req_send = comm.isend(data, dest=0, tag=num_train)
                    #end_worker_weights=time.time()
                    req = comm.irecv(source=0, tag=num_train)
                    req.wait()
                    req_send.Cancel()
                    #print('Done')

                num_train += 1

                if num_train > arglist.num_train:
                    break


            if(node_id in CENTRAL_CONTROLLER):
                received_data = []
                received_matrix = []
                while True:
                    data = comm.recv(source=MPI.ANY_SOURCE, tag=num_train)
                    received_data.append(copy.deepcopy(data))
                    received_matrix.append(H[data[0]])
                    if(matrix_rank(np.asarray(received_matrix)) >= num_agents):
                        decoded_weights = code_scheme.decode(received_data, weight_length, weight_shape)
                        #print(received_matrix)
                        if None not in decoded_weights:
                            break

                #print('enough')
                for id  in LEARNERS:
                    comm.send(1, dest=id, tag=num_train)

                # start_encoding = time.time()
                # decoded_weights = code_scheme.decode(received_data, weight_length, weight_shape)
                # end_encoding = time.time()


                for i in range(num_agents):
                    trainers[i].set_weigths(decoded_weights[i])





                num_train += 1
                if num_train > arglist.num_train:
                    break
