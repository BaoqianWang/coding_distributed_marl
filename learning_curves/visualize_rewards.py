import pickle
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect



#    Spread
with open('./simple_spreadcentralized_rewards_8_agents.pkl','rb') as f:
    centralized = pickle.load(f)

with open('./simple_spreaduncoded_rewards_8_agents.pkl','rb') as f:
    distributed = pickle.load(f)

plt.rc('font', size=18)
plt.figure(figsize=(7,4.3))
plt.plot(centralized, '-', linewidth=4, label='Original MADDPG')
plt.plot(distributed, '--', linewidth=4, label='Coded distributed MADDPG')
plt.ylabel('Reward', fontsize=18)
plt.xlabel('Training iteration (x250)', fontsize=18)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.rc('font', size=18)
plt.subplots_adjust(left=0.21, bottom=0.17, right=0.9, top=0.88, wspace=None, hspace=None)
plt.grid()
plt.savefig('spread_reward.png', transparent = True)




# #     Tag
with open('./simple_tagcentralized_rewards_8agents.pkl','rb') as f:
    centralized = pickle.load(f)

with open('./simple_taguncoded_rewards_8agents.pkl','rb') as f:
    distributed = pickle.load(f)

plt.figure(figsize=(7,4.3))
plt.plot(centralized, '-', linewidth=4, label='Original MADDPG')
plt.plot(distributed, '--', linewidth=4, label='Coded distributed MADDPG')
plt.ylabel('Reward', fontsize=18)
plt.xlabel('Training iteration (x250)', fontsize=18)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.rc('font', size=18)
plt.subplots_adjust(left=0.21, bottom=0.17, right=0.9, top=0.88, wspace=None, hspace=None)
plt.grid()
plt.savefig('tag_reward.png', transparent = True)




#     Push
with open('./simple_pushcentralized_rewards_8_agents.pkl','rb') as f:
    centralized = pickle.load(f)

with open('./simple_pushuncoded_rewards_8agents.pkl','rb') as f:
    distributed = pickle.load(f)


plt.figure(figsize=(7,4.3))
plt.plot(centralized, '-', linewidth=4, label='Original MADDPG')
plt.plot(distributed, '--', linewidth=4, label='Coded distributed MADDPG')
plt.ylabel('Reward', fontsize=18)
plt.xlabel('Training iteration (x250)', fontsize=18)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(fontsize=18)
plt.xticks(fontsize=18 )
plt.yticks(fontsize=18)
plt.rc('font', size=18)
plt.subplots_adjust(left=0.21, bottom=0.17, right=0.9, top=0.88, wspace=None, hspace=None)
plt.grid()
plt.savefig('push_reward.png', transparent = True)




#   Adversary
with open('./simple_adversarycentralized_rewards_8_agents.pkl','rb') as f:
    centralized = pickle.load(f)

with open('./simple_adversaryuncoded_rewards_8agents.pkl','rb') as f:
    distributed = pickle.load(f)

plt.figure(figsize=(7,4.3))
plt.plot(centralized, '-', linewidth=4, label='Original MADDPG')
plt.plot(distributed, '--', linewidth=4, label='Coded distributed MADDPG')
plt.ylabel('Reward', fontsize=18)
plt.xlabel('Training iteration (x250)', fontsize=18)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.rc('font', size=18)
plt.subplots_adjust(left=0.21, bottom=0.17, right=0.9, top=0.88, wspace=None, hspace=None)
#plt.show()
plt.grid()
plt.savefig('adversary_reward.png', transparent = True)
