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
import time
import json



def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=5000, help="number of episodes")
    parser.add_argument("--train-period", type=int, default=1000, help="frequency of updating parameters")
    parser.add_argument("--num_train", type=int, default=1000, help="number of train")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="maddpg", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="/home/smile/maddpg/learning_curves/", help="directory where plot data is saved")
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
    world = scenario.make_world()
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


def interact_with_environments(env, trainers, steps):

    step = 0
    obs_n=env.reset()
    episode_reward = 0

    while step < steps:
        action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        step += 1

        # collect experience
        for i, agent in enumerate(trainers):
            agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i])

        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            episode_reward += rew


    return episode_reward


if __name__=="__main__":

    #Parse the parameters
    arglist = parse_args()

    with open('/home/smile/maddpg/parameters/marl_parameters.json') as f:
        parameters = json.load(f)


    gamma = parameters['gamma']
    epsilon = parameters['epsilon']
    num_agents = parameters['num_agents']
    num_learners = parameters['uncoded_num_learners']
    reward_episode = parameters['reward_episode']
    steps = parameters['time_steps']
    max_iter = parameters['max_iteration']



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
        CENTRAL_CONTROLLER = 0
        ACTOR=1


        #Worker nodes index larger than 1



        #Initilize parameters, controllers and actors
        # if(node_id==CENTRAL_CONTROLLER):
        #     # Create environment
        #     env = make_env(arglist.scenario, arglist, arglist.benchmark)
        #     obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        #
        #     trainers = get_trainers(env, num_agents, "central", obs_shape_n, arglist,session)
        #     U.initialize()
        #     train_step=0

        #if(node_id==ACTOR):

        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        obs_n = env.reset()
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, num_agents, "actor", obs_shape_n, arglist,session)
        U.initialize()
        episode_rewards = []
        train_step=0
        iter_step=0

        if (node_id == ACTOR):
            interact_with_environments(env, trainers, arglist.batch_size * arglist.max_episode_len)



        start=time.time()
        while True:

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
            #print('Num of train %d, weights broadcast start time is' %num_train, start_master_weights)
            #print('Num of train %d, weights broadcast end time is' %num_train, end_master_weights - start_master_weights)

            #Central controller receive parameters from learners
            if(node_id==CENTRAL_CONTROLLER):

                for i in range(num_learners):
                     agent_weight= comm.recv(source=i+2, tag=num_train)
                     trainers[i].set_weigths(agent_weight)

                num_train+=1

                if num_train > arglist.num_train:
                    break

            else:
                for i, agent in enumerate(trainers):
                    agent.set_weigths(all_agents_weights[i])


            #print('Node%d' %node_id, 'Parameters set up Done', num_train)

            if(node_id==ACTOR):
                steps = 25
                env_num = 0
                while(True):
                    if(env_num == 4):
                        break
                    reward = interact_with_environments(env, trainers, steps)
                    episode_rewards.append(reward)
                    env_num += 1


                if(len(episode_rewards)%arglist.save_rate==0):
                    episode_time=time.time()
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(num_train, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),episode_time-start))
                    start=time.time()

                #print('Actor collecting number of %d episode' %(len(episode_rewards)), 'with train step', train_step)
                #if(train_step%100==0 and train_step > arglist.batch_size * arglist.max_episode_len):
                index = trainers[0].replay_buffer.make_index(arglist.batch_size)
                replay_obs_n = []
                replay_obs_next_n = []
                replay_act_n = []
                replay_reward_n = []
                replay_done_n = []
                episodes = dict()

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
                #print(episodes)

                #Send environments episodes to the learners
                for i in range(num_learners):
                    comm_start = time.time()
                    comm.send(episodes, dest=2+i, tag=num_train)
                    comm_end = time.time()
                    #print('Communication time is', comm_end - comm_start)

                num_train+=1
                #print(num_train)
                if num_train > arglist.num_train:
                    break






            if(node_id > ACTOR):

                start_recv_episode=time.time()
                episodes=comm.recv(source=1, tag=num_train)
                end_recv_episode=time.time()


                loss=None
                comp_start = time.time()
                loss=trainers[node_id-2].update(trainers, episodes)
                comp_end = time.time()

                weights = trainers[node_id-2].get_weigths()
                start_worker_weights=time.time()
                comm.send(weights, dest=0,tag=num_train)
                end_worker_weights=time.time()


                num_train+=1

                if num_train > arglist.num_train:
                    break
