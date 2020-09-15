import pickle
import matplotlib.pyplot as plt

with open('../learning_curves/maddpg_rewards.pkl','rb') as f:
    data=pickle.load(f)

plt.figure()
plt.plot(data)
plt.show()
#print(data)
