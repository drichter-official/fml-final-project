# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:13:22 2022

@author: danie
"""
import pickle
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(10,6)})

N=10

with open("rewards_dqn_task1.pt", "rb") as file:
    rewards = pickle.load(file)[:150]
    plt.plot(np.arange(len(rewards)),rewards,label="reward")
    plt.plot(np.arange(len(rewards)-N+1)+N/2,np.convolve(rewards, np.ones((N,))/N, mode='valid'),label=f"{N}-game-average")
    plt.xlabel("episode")
    plt.ylabel("rewards")
    plt.legend(loc="lower right")
    plt.savefig("rewards_dqn_task1.png", dpi=300)
    plt.show()

with open("episodelength_dqn_task1.pt", "rb") as file:
    rewards = pickle.load(file)[:150]
    plt.plot(np.arange(len(rewards)),rewards,label="reward")
    plt.plot(np.arange(len(rewards)-N+1)+N/2,np.convolve(rewards, np.ones((N,))/N, mode='valid'),label=f"{N}-game-average")
    plt.xlabel("episode")
    plt.ylabel("episodelength")
    plt.legend(loc="lower left")
    plt.savefig("episodelength_dqn_task1.png", dpi=300)
    plt.show()

N = 50

with open("rewards_qtable_task1.pt", "rb") as file:
    rewards = pickle.load(file)[:2000]
    plt.plot(np.arange(len(rewards)),rewards,label="r   eward")
    plt.plot(np.arange(len(rewards)-N+1)+N/2,np.convolve(rewards, np.ones((N,))/N, mode='valid'),label=f"{N}-game-average")
    plt.xlabel("episode")
    plt.ylabel("rewards")
    plt.legend(loc="lower left")
    plt.savefig("rewards_qtable_task1.png", dpi=300)
    plt.show()

with open("episodelength_qtable_task1.pt", "rb") as file:
    rewards = pickle.load(file)[:2000]
    plt.plot(np.arange(len(rewards)),rewards,label="reward")
    plt.plot(np.arange(len(rewards)-N+1)+N/2,np.convolve(rewards, np.ones((N,))/N, mode='valid'),label=f"{N}-game-average")
    plt.xlabel("episode")
    plt.ylabel("episodelength")
    plt.legend(loc="lower left")
    plt.savefig("episodelength_qtable_task1.png", dpi=300)
    plt.show()







with open("rewards.pt", "rb") as file:
    rewards = pickle.load(file)
    #arr = average_every(np.array(rewards),int(len(rewards)/BATCHES))
    plt.plot(np.arange(len(rewards)),rewards)
    plt.plot(np.arange(len(rewards)-N+1),np.convolve(rewards, np.ones((N,))/N, mode='valid'))
    plt.grid()
    plt.xlabel("episode")
    plt.ylabel("rewards")
    plt.title("rewards")
# =============================================================================
#     plt.savefig("rewards.pdf")
# =============================================================================
    plt.show()
    
with open("episodelength.pt", "rb") as file:
    length = pickle.load(file)
    x = length
    plt.plot(np.arange(len(x)),x)
    plt.plot(np.arange(len(length)-N+1),np.convolve(length, np.ones((N,))/N, mode='valid'))
    plt.xlabel("episode")
    plt.ylabel("episodelength")
    plt.title("episodelength")
    plt.grid()
# =============================================================================
#     plt.savefig("episodelength.pdf")
# =============================================================================
    plt.show()