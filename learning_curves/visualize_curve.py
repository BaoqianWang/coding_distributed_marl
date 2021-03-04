import pickle
import matplotlib.pyplot as plt

# with open('./simpleiteration9000velocity_6_rewards.pkl','rb') as f:
#     data1=pickle.load(f)
#
# with open('./simpleiteration5000velocity_6_rewards.pkl','rb') as f:
#     data1+=pickle.load(f)
#
# with open('./simpleiteration4000velocity_6_rewards.pkl','rb') as f:
#     data1+=pickle.load(f)


# with open('./simpleiteration10000velocity_2_rewards.pkl','rb') as f:
#     data2=pickle.load(f)
#
# with open('./simpleiteration5000velocity_2_rewards.pkl','rb') as f:
#     data2+=pickle.load(f)
#
# with open('./simpleiteration9000velocity_2_rewards.pkl','rb') as f:
#     data2+=pickle.load(f)
#
#
# with open('./simpleiteration10000velocity_3_rewards.pkl','rb') as f:
#     data3=pickle.load(f)
#
# with open('./simpleiteration5000velocity_3_rewards.pkl','rb') as f:
#     data3+=pickle.load(f)
#
# with open('./simpleiteration9000velocity_3_rewards.pkl','rb') as f:
#     data3+=pickle.load(f)
#
#
# with open('./simpleiteration10000velocity_4_rewards.pkl','rb') as f:
#     data4=pickle.load(f)
#
# with open('./simpleiteration5000velocity_4_rewards.pkl','rb') as f:
#     data4+=pickle.load(f)
#
# with open('./simpleiteration9000velocity_4_rewards.pkl','rb') as f:
#     data4+=pickle.load(f)


# plt.figure()
# plt.plot(data1, '-', label='Scenario 1')
# #plt.plot(data2, '--', label='Scenario 2')
# #plt.plot(data3, '-.', label='Scenario 3')
# #plt.plot(data4, ':', label='Scenario 4')
# plt.legend()
# plt.xlabel('Training iteration (x250)')
# plt.ylabel('Reward')
# plt.show()


# 5 agents
# with open('./simpleiteration9000velocity_6_rewards.pkl','rb') as f:
#     data2=pickle.load(f)
#
# with open('./simpleiteration5000velocity_6_rewards.pkl','rb') as f:
#     data2+=pickle.load(f)
#
# with open('./simpleiteration4000velocity_6_rewards.pkl','rb') as f:
#     data2+=pickle.load(f)
#
# with open('./simpleiteration8000velocity_6_rewards.pkl','rb') as f:
#     data2+=pickle.load(f)
#
#
#
# # 3 agents
# with open('./simpleiteration4000velocity_5_rewards.pkl','rb') as f:
#     data1=pickle.load(f)
#
# with open('./simpleiteration10000velocity_5_rewards.pkl','rb') as f:
#     data1+=pickle.load(f)
#
# with open('./simpleiteration8000velocity_5_rewards.pkl','rb') as f:
#     data1+=pickle.load(f)
#
# with open('./simpleiteration5000velocity_5_rewards.pkl','rb') as f:
#     data1+=pickle.load(f)



# 4 agents
# with open('./simpleiteration18000velocity_7_rewards.pkl','rb') as f:
#     data3=pickle.load(f)
#
# with open('./simpleiteration8000velocity_7_rewards.pkl','rb') as f:
#     data3_1 =pickle.load(f)
#
# with open('./simpleiteration7000velocity_7_rewards.pkl','rb') as f:
#     data3_2 =pickle.load(f)
#
#
# data3 = data3[3:] + data3_1[2:] + data3_2[:-4]
# plt.figure()
# plt.plot(data1[1:-3], '-', label='Scenario 1')
# plt.plot(data2[3:], '-', label='Scenario 1')
# plt.legend()
# plt.xlabel('Training iteration (x250)')
# plt.ylabel('Reward')
# plt.show()

with open('./simple_spreadneighbor_benchmark_rewards_3_agents.pkl','rb') as f:
    data1=pickle.load(f)

# with open('./simple_spreadneighbor_rewards_3_agents_1000.pkl','rb') as f:
#     data2=pickle.load(f)

plt.figure(figsize=(8,5.3))
plt.plot(data1, '--', linewidth=4, label='Scenario 1')
plt.ylabel('Reward', fontsize=18)
plt.xlabel('Training iteration (x100)', fontsize=18)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(fontsize=18)
plt.xticks(fontsize=18 )
plt.yticks(fontsize=18)
plt.grid()
#plt.show()
plt.savefig('benchmark_neighbor_3.png', transparent = True)


#print(data)
