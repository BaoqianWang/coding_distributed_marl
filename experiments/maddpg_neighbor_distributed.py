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
    parser.add_argument("--plots-dir", type=str, default="/home/smile/maddpg/learning_curves/distributed/", help="directory where plot data is saved")
    parser.add_argument("--num_straggler", type=int, default="0", help="num straggler")
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


def interact_with_environments(env, trainers, node_id, steps):

    for i in range(steps):
        obs_neighbor = env.reset(node_id) # Only initalize neighbor area of nodei_id-th agent
        action_n = [[]] * env.n
        target_action_n = [[]] * env.n
        for i, obs in enumerate(obs_neighbor):
            if len(obs) !=0:
                action_n[i] = trainers[i].action(obs)

        new_obs_neighbor, rew, done_n, next_info_n = env.step(action_n) # Interaction within the neighbor area

        for j, next_obs in enumerate(new_obs_neighbor):
            if len(obs) !=0:
                target_action_n[i] = trainers[i].target_action(obs)

        info_n = 0.1
        ## Information needed:
        # Observation of node_id, actions of nearby agents, Observations of nearby agents at next time steps:
        trainers[node_id].experience(obs_neighbor[node_id], action_n, new_obs_neighbor[node_id], target_action_n, rew) # Add one step transition to replay memory of node_id -th agent
        print('observation', obs_neighbor, 'reward', rew, 'action', action_n, 'new_observation', new_obs_neighbor, info_n)
    return episode_rewards


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


        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        num_agents = env.n
        num_learners = num_agents
        LEARNERS = [i for i in range(1, 1+num_learners)]

        assert num_node == num_learners + 1


        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, num_agents, "actor", obs_shape_n, arglist, session)
        U.initialize()
        episode_rewards = []
        episodes = []
        final_episode_rewards = []
        train_time = []
        train_step=0
        iter_step=0

        if (node_id == CENTRAL_CONTROLLER):
            print('Coded computation scheme: ', 'Uncoded')
            print('Scenario: ', arglist.scenario)
            print('Number of learners: ', num_learners)
            print('Number of agents: ', num_agents)

            #interact_with_environments(env, trainers, steps, True)

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


            if(node_id in LEARNERS):
                # Receive parameters
                for i, agent in enumerate(trainers):
                    agent.set_weigths(all_agents_weights[i])


                interact_with_environments(env, trainers, node_id-1, steps)

                # Interact with environment in neighbor area

                # Build the replay memory for node_id-th agent

                #


                # Update parameters
                # loss=None
                # comp_start = time.time()
                # loss=trainers[node_id-1].update(trainers, episodes)
                # comp_end = time.time()
                #
                # # Send parameters to central controller
                # weights = trainers[node_id-1].get_weigths()
                #
                # if(node_id in STRAGGLER):
                #     time.sleep(0.25)
                #
                # start_worker_weights=time.time()
                # comm.send(weights, dest=0,tag=num_train)
                # end_worker_weights=time.time()

                num_train+=1
                if num_train > arglist.num_train:
                    break

            # Central controller receives weights from learners
            if(node_id == CENTRAL_CONTROLLER):
                for i in range(num_learners):
                     agent_weight = comm.recv(source=i+1, tag=num_train)
                     trainers[i].set_weigths(agent_weight)

                num_train += 1
                if num_train > arglist.num_train:
                    break
