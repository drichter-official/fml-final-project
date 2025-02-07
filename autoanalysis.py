# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 23:58:15 2022

@author: danie
"""
import pickle
import os
import sys
import gc

args = [
    "main.py",
    "play",
    "--agents",
    "bestcp",
    "--no-gui",
    "--n-rounds",
    "10",
    "--train",
    "1"
]
# ['ckpt-1-rew_56.6', 'checkpoint-eps_5600-rew_69.5-1','checkpoint-eps_600-rew_48.3-1', 'checkpoint-eps_9800-rew_48.0-1', 'ckpt-2-rew_62.4', 'checkpoint-eps_4800-rew_51.9-1', 'checkpoint-eps_12200-rew_53.7-1', 'checkpoint-eps_5000-rew_59.0-1']

k = 2
sys.argv = args
command_str = "python "
for s in args:
    command_str += s+" "
#print(command_str)
# =============================================================================
# path = "C:/Users/danie/Meine Ablage/Github/bomberman_rl/agent_code/bestcp/Checkpoints/"
# for i in range(len([a+b for a,b in [f.split(".")[0:2] for f in os.listdir(path)]])):
#         os.environ["AUTOANALYSIS"] = "YES"
#         k = i
#         res = os.system(command_str)
#         print(res)
#         if res == 0:
#             with open("coins.pt", "rb") as file:
#                 rewards = pickle.load(file)
#                 #arr = average_every(np.array(rewards),int(len(rewards)/BATCHES))
#                 print(rewards)
#                 
#             with open("score.pt", "rb") as file:
#                 score = pickle.load(file)
#                 #arr = average_every(np.array(rewards),int(len(rewards)/BATCHES))
#                 print(sum(score))
#             gc.collect()
#         else:
#             raise
# 
# os.environ["AUTOANALYSIS"] = "NO"
# =============================================================================
