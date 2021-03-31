import pickle
import matplotlib.pyplot as plt

with open('../learning_curves/simple_spreadneighbor_distributed_num_agents_3.pkl','rb') as f:
    data=pickle.load(f)

plt.figure(figsize=(8,5.3))
plt.plot(data, linewidth=4)
plt.ylabel('Reward', fontsize=18)
plt.xlabel('Training iteration (x50)', fontsize=18)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
#plt.savefig('3_agents_comparison_without_goal.png', transparent = True)
plt.show()
#print(data)
