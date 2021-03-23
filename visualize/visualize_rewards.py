import pickle
import matplotlib.pyplot as plt

with open('../learning_curves/simple_spreadneighbor_distributed_num_agents_3.pkl','rb') as f:
    data=pickle.load(f)

plt.figure()
plt.plot(data)
plt.show()
#print(data)
