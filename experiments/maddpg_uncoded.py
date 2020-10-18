import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
from mpi4py import MPI
import random
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg_coding_replay_memory import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import json
from experiments.common_configuration import parse_args, mlp_model, make_env

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



if __name__== "__main__":

    #Parse the parameters
    arglist = parse_args()


    gamma = arglist.gamma
    steps = arglist.max_episode_len
    num_straggler = arglist.num_straggler


    with tf.Session() as session:
        num_train=0

        tf.set_random_seed(30)
        random.seed(30)

        # MPI initialization.
        comm = MPI.COMM_WORLD
        num_node = comm.Get_size()

        node_id = comm.Get_rank()
        node_name = MPI.Get_processor_name()
        CENTRAL_CONTROLLER = 0
        ACTOR = 1


        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        num_agents = env.n
        num_learners = num_agents
        LEARNERS = [i for i in range(2, 2+num_learners)]

        assert num_node == num_learners + 2

        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, num_agents, "actor", obs_shape_n, arglist, session)
        U.initialize()
        episode_rewards = []
        episodes = []
        final_episode_rewards = []
        train_time = []
        train_step=0
        iter_step=0

        if (node_id == ACTOR):
            print('Coded computation scheme: ', 'Uncoded')
            print('Scenario: ', arglist.scenario)
            print('Number of learners: ', num_learners)
            print('Number of agents: ', num_agents)
            interact_with_environments(env, trainers, steps, True)

        done = 0


        start = time.time()
        while True:
            STRAGGLER = random.sample(LEARNERS, num_straggler)

            #Central controller broadcast weights
            if(node_id==CENTRAL_CONTROLLER):
                all_agents_weights=[]
                for i, agent in enumerate(trainers):
                    all_agents_weights.append(agent.get_weigths())
            else:
                all_agents_weights=None


            start_master_weights=time.time()
            all_agents_weights=comm.bcast(all_agents_weights,root=0)
            end_master_weights=time.time()


            if(node_id > CENTRAL_CONTROLLER):
                for i, agent in enumerate(trainers):
                    agent.set_weigths(all_agents_weights[i])


            if(node_id == ACTOR):
                rewards = interact_with_environments(env, trainers, 100, False)
                rewards.pop()
                #print('rewards is', rewards)
                episode_rewards += rewards

                #print(num_train)
                if(len(episode_rewards)%arglist.save_rate==0):
                    episode_time=time.time()
                    train_time.append(episode_time-start)
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(num_train, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),episode_time-start))
                    final_episode_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                    start=time.time()

                #print('Actor collecting number of %d episode' %(len(episode_rewards)), 'with train step', train_step)
                #if(train_step%100==0 and train_step > arglist.batch_size * arglist.max_episode_len):
                #for l in range(num_learners):
                replay_obs_n = []
                replay_obs_next_n = []
                replay_act_n = []
                replay_reward_n = []
                replay_done_n = []
                episodes = dict()

                index = trainers[0].replay_buffer.make_index(arglist.batch_size)
                for i in range(num_learners):
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

            episodes = comm.bcast(episodes, root=1)


            if(node_id == ACTOR):
                num_train += 1
                if num_train > arglist.num_train:
                    print('The mean time is', np.mean(train_time[1:]), 'The corresponding variance is', np.var(train_time[1:]))
                    rew_file_name = arglist.plots_dir + arglist.scenario + 'uncoded_rewards_%d_agents.pkl' %arglist.num_agents
                    with open(rew_file_name, 'wb') as fp:
                        pickle.dump(final_episode_rewards, fp)
                    break


            if(node_id > ACTOR):
                loss=None
                #episodes = comm.recv(source=1, tag=num_train)
                comp_start = time.time()
                loss=trainers[node_id-2].update(trainers, episodes)
                comp_end = time.time()
                weights = trainers[node_id-2].get_weigths()

                if(node_id in STRAGGLER):
                    time.sleep(0.25)

                start_worker_weights=time.time()
                comm.send(weights, dest=0,tag=num_train)
                end_worker_weights=time.time()

                num_train+=1
                if num_train > arglist.num_train:
                    break


            if(node_id == CENTRAL_CONTROLLER):
                for i in range(num_learners):
                     agent_weight = comm.recv(source=i+2, tag=num_train)
                     trainers[i].set_weigths(agent_weight)

                num_train += 1
                if num_train > arglist.num_train:
                    break
