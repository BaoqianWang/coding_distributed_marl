trainers = get_trainers(env, num_agents, "learner", obs_shape_n, arglist,session)
U.initialize()
obs_n = env.reset()
train_step=0
while True:
    #Interaction step
    iter_step=0
    #Interact with environment to get experience

    while True:

        # get action
        action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        train_step+=1
        iter_step+=1
        done = all(done_n)
        terminal=(iter_step>=arglist.max_episode_len)

        # collect experience
        for i, agent in enumerate(trainers):
            agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i])

        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew

        if done or terminal:
            obs_n=env.reset()
            episode_rewards.append(0)
            break

    loss=None
    for agent in trainers:
        agent.preupdate()



    if(train_step%200==0):
        for agent in trainers:
            loss=agent.update(trainers)

        for agent in trainers:
            agent.replay_buffer.clear()


        # if(iter_step>arglist.batch_size):
        #     break
    if (len(episode_rewards) % arglist.save_rate == 0):
        print("steps: {}, episodes: {}, mean episode reward: {}".format(train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:])))

    if len(episode_rewards) > arglist.num_episodes:
        break
