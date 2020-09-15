import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
from mpi4py import MPI
import random
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg_coding import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import time





def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=5000, help="number of episodes")
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

if __name__=="__main__":

    #Parse the parameters
    arglist = parse_args()
    with tf.Session() as session:
        num_train=0
        pre_num_train=0
        tf.set_random_seed(30)
        random.seed(30)
        # MPI initialization.
        comm = MPI.COMM_WORLD
        num_node = comm.Get_size()
        node_id = comm.Get_rank()
        node_name = MPI.Get_processor_name()
        CENTRAL_CONTROLLER = 0

        num_learners=10 # This is determined in the environment
        num_agents=10


        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        obs_n=env.reset()
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]

        trainers = get_trainers(env, num_agents, "learner", obs_shape_n, arglist,session)
        U.initialize()
        train_step=0
        iter_step=0
        episode_rewards = [0.0]

        start_time=time.time()
        while True:

            iter_step=0
            start_env=time.time()
            while True:
                action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
                # environment step
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                train_step+=1
                iter_step+=1
                done = all(done_n)
                terminal=(iter_step>=arglist.max_episode_len)

                # collect experience
                for i, agent in enumerate(trainers):
                    agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i])

                obs_n = new_obs_n

                for i, rew in enumerate(rew_n):
                    episode_rewards[-1] += rew

                if done or terminal:
                    obs_n=env.reset()
                    episode_rewards.append(0)
                    break
            end_env=time.time()
            #print("env_time_is",end_env-start_env)
            loss=None
            for agent in trainers:
                agent.preupdate()

            if(train_step%100==0):
                start=time.time()
                for agent in trainers:
                     loss=agent.update(trainers)
                #trainers[0].update(trainers)
                end=time.time()

                #print("train_time_is",end-start)
                #for agent in trainers:
                    #agent.replay_buffer.clear()


            if (len(episode_rewards) % arglist.save_rate == 0):
                episode_time=time.time()
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),episode_time-start_time))
                start_time=time.time()

            if len(episode_rewards) > arglist.num_episodes:
                break
