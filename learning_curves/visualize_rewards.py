import pickle
import matplotlib.pyplot as plt

with open('./simple_spreadcentralized_rewards_8_agents.pkl','rb') as f:
    centralized = pickle.load(f)

with open('./simple_spreaduncoded_rewards_8_agents.pkl','rb') as f:
    distributed = pickle.load(f)




plt.figure()
plt.plot(centralized, label='Centralized')
plt.plot(distributed, label='Distributed')
plt.ylabel('Reward', fontsize=14)
plt.xlabel('Training iteration (x250)', fontsize=14)
plt.legend(fontsize=14)
plt.xticks(fontsize=14 )
plt.yticks(fontsize=14)
plt.show()
