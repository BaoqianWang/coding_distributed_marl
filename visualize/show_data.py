import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pickle



scenario = ['simple', 'simple_adversary', 'simple_crypto', 'simple_push', 'simple_spread', 'simple_tag', 'simple_speaker_listener']

for i, name in enumerate(scenario):
    data = np.loadtxt('./regular/%s_reward_good' %name)
    data1 = np.loadtxt('./distributed/%s_dis_reward_good' %name)



    plt.figure(i)
    plt.plot(data[1:], label=' MADDPG')
    plt.plot(data1, label='Distributed MADDPG')


    with open('./POMDP/%s.pkl' %name,'rb') as f:
        data2=pickle.load(f)


    plt.plot(data2[1:], label='Policy Gradient')




    plt.xlabel('Number of iterations (x250)')
    plt.ylabel('Rewards')
    plt.title(name)
    plt.legend()
    plt.show()


for i, name in enumerate(scenario):
    data = np.loadtxt('./regular/%s_time' %name)
    data1 = np.loadtxt('./distributed/%s_dis_time' %name)

    print('MADDPG %s' %name, np.mean(data[1:]))
    print('Distributed MADDPG %s' %name, np.mean(data1))


    # plt.figure(i)
    # plt.plot(data[1:], label=' MADDPG')
    # plt.plot(data1, label='Distributed MADDPG')
    #
    #
    # with open('./POMDP/%s.pkl' %name,'rb') as f:
    #     data2=pickle.load(f)
    #
    #
    # plt.plot(data2[1:])
    #
    #
    #
    #
    # plt.xlabel('Number of iterations (x1000)')
    # plt.ylabel('Rewards')
    # plt.title(name)
    # plt.legend()
    # plt.show()


#print(data)
