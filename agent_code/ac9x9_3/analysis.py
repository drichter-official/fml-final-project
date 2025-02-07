# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 04:55:09 2022

@author: ztec1
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid')/w
def average_every(x, w):
    arr = np.nanmean(np.pad(x.astype(float), ( 0, 0 if x.size % w == 0 else w - x.size % w ), mode='constant', constant_values=np.NaN).reshape(-1, w), axis=1)
    return arr
    
BATCHES = 10
N=100

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
    
    
    