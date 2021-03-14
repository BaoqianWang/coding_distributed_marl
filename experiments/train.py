import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import random
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from experiments.common_configuration import parse_args, mlp_model, make_env, get_trainers
import imageio




def interact_with_environments(env, trainers, size_transitions, train_data = True):

    obs_n = env.reset()
    episode_rewards = [0.0]
    step = 0
    num_transitions = 0
    while True:

        action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        step += 1
        done = all(done_n)
        terminal = (step >= arglist.max_episode_len)
        # collect experience
        if (train_data):
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i])

        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
        if done or terminal:
            obs_n = env.reset()
            episode_rewards.append(0)
            step = 0

        num_transitions += 1

        if num_transitions >= size_transitions:
            break

        # if len(trainers[0].replay_buffer) >= arglist.max_episode_len * arglist.batch_size:
        #     break

    return np.mean(episode_rewards)

def train(arglist):
    with U.single_threaded_session():
        tf.set_random_seed(30)
        random.seed(30)
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, 1)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        train_time = []
        num_train = 0
        print('Coded computation scheme: ', 'Centralized Training')
        print('Scenario: ', arglist.scenario)
        print('Number of agents: ', arglist.num_agents)
        print('Starting iterations...')

        #Collect enough data for memory
        if not arglist.display:
            env_time1 = time.time()
            interact_with_environments(env, trainers, 5 * arglist.batch_size)
            env_time2 = time.time()
            print('Environment interactation time: ', env_time2 - env_time1)

        t_start = time.time()
        k = 1
        frames = []
        computation_time = []
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            #print(action_n)
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i])
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1
            #print(train_step)
            if (train_step % 100 == 0):
                if(arglist.num_straggler):
                    time.sleep(1)
                num_train += 1

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                #env.render()
                k+=1
                frames.append(env.render('rgb_array')[0])
                if (terminal or done):
                    imageio.mimsave('real_benchmark_neighbor%d_agents%d.gif' %(k,arglist.num_agents), frames, duration=0.15)
                    frames=[]

                if( k>= 200):
                    #imageio.mimsave('test.gif', frames, duration=1)
                    break
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()

            comp_time1 = time.time()
            for agent in trainers:
                loss = agent.update(trainers, train_step)
            comp_time2 = time.time()
            #print('Computation time:', comp_time2 - comp_time1)
            computation_time.append(comp_time2 - comp_time1)


            # save model, display training output
            if (num_train %100 ==0 and train_step %100 ==0):
                #print(num_train)
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                reward = interact_with_environments(env, trainers, 3*arglist.max_episode_len, False)
                t_end = time.time()
                print("steps: {},  mean episode reward: {}, time: {}".format(
                    num_train, reward, round(t_end-t_start, 3)))
                print('Computation time:', np.max(computation_time))
                computation_time = []
                train_time.append(round(t_end-t_start, 3))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(reward)


            # saves final episode reward for plotting training curve later
            if num_train > arglist.max_num_train:
                print('The mean time is', np.mean(train_time[1:]), 'The corresponding variance is', np.var(train_time[1:]))
                rew_file_name = arglist.plots_dir + arglist.scenario + 'neighbor_benchmark_rewards_%d_agents.pkl' %arglist.num_agents
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                break



if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
