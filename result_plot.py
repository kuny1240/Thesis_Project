import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

total_results = []
losses = []
episode = []

initial_num = 999

for i in range(13):
    episode.append(i*1000+initial_num)
    losses.append(np.loadtxt('./Results/losses{}.txt'.format(i*1000 + initial_num)))
    total_results.append(np.loadtxt('./Results/rewards{}.txt'.format(i*1000 + initial_num)))



losses = np.vstack(losses)
total_results = np.vstack(total_results)


avg_losses = np.mean(losses,axis=1)


avg_reward = np.mean(total_results,axis=1)
max_reward = np.max(total_results,axis=1)
min_reward = np.min(total_results,axis=1)
reward_num = np.sum(total_results > 0,axis=1)


plt.figure(1)
plt.plot(episode,avg_losses)
plt.title("Average Losses Over episodes")
plt.xlabel("Episode Number")
plt.ylabel("Average Losses")
plt.legend("avg_losses")

plt.show()


plt.figure(2)
plt.plot(episode,avg_reward,label="avg_reward")
plt.plot(episode,max_reward,label="max_reward")
# plt.plot(episode,min_reward,label="min_reward")

plt.title("Average reward Over episodes")
plt.xlabel("Episode Number")
plt.ylabel("Average Reward")
plt.legend(["avg_reward","max_reward"])

plt.show()

plt.figure(3)
plt.plot(episode,reward_num/1000,label="improve_rate")

plt.show()