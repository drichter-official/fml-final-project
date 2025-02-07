# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 04:55:09 2022

@author: ztec1
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

N=10

def moving_average(x):
    return np.convolve(x, np.ones((N,))/N, 'valid')

    
with open("rewards.pt", "rb") as file:
    rewards = pickle.load(file)
    plt.plot(np.arange(len(rewards)),rewards)
    plt.plot(np.arange(len(rewards)-N+1),moving_average(rewards))
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
    
    epsx = len(length)
    y= [1]
    for i in range(epsx-1):
        y.append(0.99**i)
    
    plt.plot(np.arange(epsx),np.array(y)*400)  
    
    plt.xlabel("episode")
    plt.ylabel("episodelength")
    plt.title("episodelength")
# =============================================================================
#     plt.savefig("episodelength.png")
# =============================================================================
    plt.show()
    