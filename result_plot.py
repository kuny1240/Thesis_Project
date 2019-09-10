import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Result_Folders = ["No_Noise_Result","Small_Noise_Result","Large_Noise_Result"]
colors = ['r','b','y']



f0 = plt.figure(0)
plt.title("Training Loss Over Episodes")
plt.xlabel("Episode Number")
plt.ylabel("Average Losses")

initial_num = 999

for j,F in enumerate(Result_Folders):
    total_results = []
    losses = []
    episode = []
    val_rewards = []
    for i in range(10):
        episode.append(i*1000+initial_num)
        losses.append(np.loadtxt('./Results/' + F + '/losses{}.txt'.format(i*1000 + initial_num)))
        total_results.append(np.loadtxt('./Results/'+F+'/rewards{}.txt'.
                                        format(i*1000 + initial_num)))
        val_rewards.append(np.loadtxt('./Results/'+F+'/rewards_val{}.txt'.
                                        format(i * 1000 + initial_num)))

    losses = np.vstack(losses)
    total_results = np.vstack(total_results)
    val_rewards = np.vstack(val_rewards)

    avg_losses = np.mean(losses, axis=1)
    avg_reward = np.mean(total_results, axis=1)
    avg_val_reward = np.mean(val_rewards, axis=1)
    reward_num = np.sum(total_results > 5, axis=1)
    val_rewards_num = np.sum(val_rewards > 5, axis=1)

    plt.figure(0)
    plt.plot(episode, avg_losses,colors[j], label=F + " avg_losses" )
    plt.legend()

    plt.figure(2)
    plt.plot(episode, avg_reward,colors[j] + '-',label=F+" avg_reward")
    plt.plot(episode, avg_val_reward, colors[j] + '--', label=F+" val_avg_reward")
    # plt.plot(episode,max_reward,label="max_reward")
    # plt.plot(episode,min_reward,label="min_reward")

    plt.title("Average reward Over episodes")
    plt.xlabel("Episode Number")
    plt.ylabel("Average Reward")
    plt.legend()

    plt.figure(3)
    plt.plot(episode, reward_num / 1000,colors[j] + '-', label=F +" improve_rate")
    plt.plot(episode, val_rewards_num / 1000,colors[j] + '--' ,label=F+" val_improve_rate")
    plt.title("Average improving rate Over episodes")
    plt.xlabel("Episode Number")
    plt.ylabel("Average Improving Rate")
    plt.legend()





plt.show()





