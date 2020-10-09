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


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=20000, help="number of episodes")
    parser.add_argument("--train-period", type=int, default=1000, help="frequency of updating parameters")
    parser.add_argument("--num_train", type=int, default=50, help="number of train")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    parser.add_argument("--scheme", type=str, default="MDS", help="computation schemes including replication-based, mds, ...")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="maddpg", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=20, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="/home/smile/maddpg/learning_curves/", help="directory where plot data is saved")
    parser.add_argument("--num_straggler", type=int, default="0", help="num straggler")
    parser.add_argument("--num_learners", type=int, default="0", help="num learners")
    parser.add_argument("--vanLDPC_p", type=int, default="0", help="LDPC_p")
    parser.add_argument("--vanLDPC_pho", type=int, default="0", help="LDPC_pho")
    parser.add_argument("--vanLDPC_gamma", type=int, default="0", help="LDPC_gamma")
    parser.add_argument("--num_agents", type=int, default="0", help="num agents")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(arglist.num_agents)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_agents, name, obs_shape_n, arglist, session):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_agents):
        trainers.append(trainer(
            name+"agent_%d" % i, model, obs_shape_n, session,env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))

    return trainers


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
                episodes_total = []
                for l in range(num_agents):
                    replay_obs_n = []
                    replay_obs_next_n = []
                    replay_act_n = []
                    replay_reward_n = []
                    replay_done_n = []
                    episodes = dict()

                    index = trainers[l].replay_buffer.make_index(arglist.batch_size)
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
                    episodes_total.append(episodes)

                for l, row in enumerate(H):
                    episodes_total_send = [None]*num_agents
                    for j, entry in enumerate(row):
                        if entry != 0:
                            episodes_total_send[j] = episodes_total[j]

                    comm.send(episodes_total_send, dest=2+l, tag=num_train)


            if(node_id in ACTOR):
                print(num_train)
                num_train += 1
                #print(num_train)
                if num_train > arglist.num_train:
                    print('The mean time is', np.mean(train_time), 'The corresponding variance is', np.var(train_time))
                    break


            if(node_id in LEARNERS):
                loss = None
                episodes_total = comm.recv(source=1, tag=num_train)
                weights = [None] * num_agents

                for i, entry in enumerate(H[node_id-2]):
                    if (entry != 0):
                        episodes = episodes_total[i]
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
                        if(clock >= .25):
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
                        break

                #print('enough')
                for id  in LEARNERS:
                    comm.send(1, dest=id, tag=num_train)

                start_encoding = time.time()
                decoded_weights = code_scheme.decode(received_data, weight_length, weight_shape)
                end_encoding = time.time()


                for i in range(num_agents):
                    trainers[i].set_weigths(decoded_weights[i])





                num_train += 1
                if num_train > arglist.num_train:
                    break
