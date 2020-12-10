import pickle


final_rewards = [-105.73, -58.46, -56.00, -55.86, -56.46, -56.69, -55.79, -55.01, -55.82, -55.87, -54.75, -54.42, -54.79, -55.51, -54.85]
rew_file_name = 'simple_pushuncoded_rewards_8agents.pkl'


with open(rew_file_name, 'wb') as fp:
    pickle.dump(final_rewards, fp)
