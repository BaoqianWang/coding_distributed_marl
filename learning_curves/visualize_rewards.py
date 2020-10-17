import pickle
import matplotlib.pyplot as plt

with open('./simple_spreadcentralized_rewards_3_agents.pkl','rb') as f:
    data=pickle.load(f)

plt.figure()
plt.plot(data)
plt.ylabel('Reward')
plt.xlabel('Training iteration')
plt.show()
