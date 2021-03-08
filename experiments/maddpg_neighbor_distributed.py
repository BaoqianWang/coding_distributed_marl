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
import imageio



def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=20000, help="number of episodes")
    parser.add_argument("--train-period", type=int, default=1000, help="frequency of updating parameters")
    parser.add_argument("--num_train", type=int, default=2000, help="number of train")
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
    parser.add_argument("--save-dir", type=str, default="/home/smile/maddpg/trained_policy/", help="directory in which training state and model should be saved")
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

def make_env(scenario_name, arglist, evaluate=False): ###################
    import multiagent.scenarios as scenarios
    if evaluate:
        from multiagent.environment_neighbor_evaluate import MultiAgentEnv
        scenario = scenarios.load(scenario_name + "_neighbor_evaluate.py").Scenario()
    else:
        from multiagent.environment import MultiAgentEnv
        scenario = scenarios.load(scenario_name + ".py").Scenario()


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

def evaluate_policy(evaluate_env, trainers, display = False):

    episode_rewards = [0.0]
    step = 0
    num_evaluation = 3
    num_transitions = 0
    frames = []
    obs_n, info_n = evaluate_env.reset()
    while True:
        action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]

        new_obs_n, rew_n, done_n, next_info_n = evaluate_env.step(action_n)
        num_transitions+=1
        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
        step += 1
        done = all(done_n)
        terminal = (step >= arglist.max_episode_len)

        obs_n = new_obs_n
        info_n = next_info_n


        if arglist.display:
            time.sleep(0.1)
            frames.append(evaluate_env.render('rgb_array')[0])
            if (terminal or done):
                imageio.mimsave('demo_num_agents_%d_%d.gif' %(arglist.num_agents, num_transitions), frames, duration=0.15)
                frames=[]

        if done or terminal:
            episode_rewards.append(0)
            obs_n, info_n = evaluate_env.reset()
            step = 0

        if num_transitions >= 3*arglist.max_episode_len :
            break

    return np.mean(episode_rewards)


def interact_with_environments(env, trainers, node_id, steps):

    for i in range(steps):
        obs_neighbor = env.reset(node_id) # Only initalize neighbor area of nodei_id-th agent
        action_n = [np.array([0,0,0,0,0])] * env.n
        target_action_n = [np.array([0,0,0,0,0])] * env.n
        for i, obs in enumerate(obs_neighbor):
            if len(obs) !=0:
                action_n[i] = trainers[i].action(obs)

        new_obs_neighbor, rew, done_n, next_info_n = env.step(action_n) # Interaction within the neighbor area

        for j, next_obs in enumerate(new_obs_neighbor):
            if len(next_obs) !=0:
                target_action_n[i] = trainers[i].target_action(next_obs)
        #print(target_action_n)
        info_n = 0.1
        ## Information needed:
        # Observation of node_id, actions of nearby agents, Observations of nearby agents at next time steps:
        trainers[node_id].experience(obs_neighbor[node_id], action_n, new_obs_neighbor[node_id], target_action_n, rew) # Add one step transition to replay memory of node_id -th agent
        #print('observation', obs_neighbor, 'reward', rew, 'action', action_n, 'new_observation', new_obs_neighbor, info_n)
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


        env = make_env(arglist.scenario, arglist, evaluate= False)
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
            evaluate_env = make_env(arglist.scenario, arglist, evaluate= True)
            saver = tf.train.Saver()
            final_rewards = []

            #interact_with_environments(env, trainers, steps, True)

        done = 0

        start_time = time.time()

        if(arglist.display):
            if(node_id == CENTRAL_CONTROLLER):
                print('Loading previous state...')
                U.load_state(arglist.save_dir)
                evaluate_policy(evaluate_env, trainers, display = True)
        else:
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

                    if (num_train == 0):
                        interact_with_environments(env, trainers, node_id-1, 5*arglist.batch_size)
                    else:
                        interact_with_environments(env, trainers, node_id-1, 4*steps)

                    loss=trainers[node_id-1].update(trainers)
                    weights = trainers[node_id-1].get_weigths()
                    comm.send(weights, dest=0,tag=num_train)
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
                    #print('Num of iteration', num_train)
                    if(num_train % 100 == 0):
                        end_train_time = time.time()
                        U.save_state(arglist.save_dir, saver=saver)
                        rew_evaluate = evaluate_policy(evaluate_env, trainers)
                        final_rewards.append(rew_evaluate)
                        print('Num of training iteration:', num_train, 'Reward:', rew_evaluate, 'Training time:', end_train_time - start_time)
                        start_time = time.time()

                    if num_train > arglist.num_train:
                        rew_file_name = arglist.plots_dir + arglist.scenario + '_num_agents_%d.pkl' %arglist.num_agents
                        with open(rew_file_name, 'wb') as fp:
                            pickle.dump(final_rewards, fp)
                        break
