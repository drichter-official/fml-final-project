# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 04:55:09 2022

@author: ztec1
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

N=100

with open("rewards.pt", "rb") as file:
    rewards = pickle.load(file)
    plt.plot(np.arange(len(rewards)),rewards)
    plt.plot(np.arange(len(rewards)-N+1),np.convolve(rewards, np.ones((N,))/N, mode='valid'))
    plt.grid()
    plt.xlabel("episode")
    plt.ylabel("rewards")
    plt.title("rewards")
# =============================================================================
#     plt.savefig("rewards.png")
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
#     plt.savefig("episodelength.png")
# =============================================================================
    plt.show()
    
    
    