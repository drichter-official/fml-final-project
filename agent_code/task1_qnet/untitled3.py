# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 00:48:18 2022

@author: ztec1
"""
import numpy as np
import tensorflow as tf
field = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1],
       [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        -1],
       [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0,
        -1],
       [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        -1],
       [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0,
        -1],
       [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        -1],
       [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0,
        -1],
       [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        -1],
       [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0,
        -1],
       [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        -1],
       [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0,
        -1],
       [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        -1],
       [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0,
        -1],
       [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        -1],
       [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0,
        -1],
       [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        -1],
       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1]])

crates = np.array(np.where((field==0)))

ownpos = np.array([1,3])
print(np.argmin(np.linalg.norm(crates-ownpos[:,None],axis=0)))
if crates.size>0:
    print('success')
else: 
    print('failure')
pos = np.array([9,10])
cratepos = crates[:,np.argmin(np.linalg.norm(crates -ownpos[:,None],axis=0))]
print(cratepos)
# =============================================================================
# coin_field  = np.zeros(shape=field.shape)
coins = [(11, 2), (11, 9), (12, 15), (8, 15), (15, 14), (11, 8), (7, 12), (9, 3), (3, 5), (10, 1), (12, 13), (1, 10), (13, 9), (9, 4), (2, 15), (8, 13), (14, 15), (14, 1), (13, 14), (5, 15), (15, 5), (13, 2), (10, 11), (8, 3), (15, 6), (1, 11), (9, 2), (13, 11), (13, 4), (5, 2), (2, 7), (8, 9), (5, 3), (4, 11), (4, 3), (15, 1), (9, 7), (1, 5), (9, 13), (13, 15), (3, 15), (13, 8), (1, 8), (12, 11), (11, 13), (3, 12)]
# for coin in coins:
#     coin_field.itemset(coin,1)
# tensor = tf.convert_to_tensor([field.flatten(),coin_field.flatten()])
# #print(tensor)
# surround = np.zeros(shape=(3,3))
# ownpos = [1,3]
# surround = field[ownpos[0]-1:ownpos[0]+2,ownpos[1]-1:ownpos[1]+2]
# print(surround)
# # =============================================================================
# # for x in range(3):
# #     for y in range(3):
# #         surround.itemset(np.array([x,y])+ownpos,field[x+ownpos[0],y+ownpos[1]])
# # =============================================================================
# nmean = np.mean(coins,axis=0)
# print(coins[np.argmin(np.array(coins)-np.array(ownpos))])
# # =============================================================================
# # trans = ("Transition", ("state", "action", "next_state", "reward"))
# # print(trans.Transition)
# # =============================================================================
# ####
# 
# coins = [(7, 10), (6, 11), (10, 15), (5, 11), (9, 4), (12, 15), (8, 13), (15, 14), (9, 15), (14, 15), (6, 3), (9, 10), (2, 11), (14, 1), (15, 7), (14, 9), (2, 13), (5, 6), (15, 15), (14, 5), (15, 2), (3, 8), (13, 3), (5, 2), (13, 1), (3, 13), (9, 14), (15, 3), (7, 5), (5, 1), (1, 6), (6, 1), (6, 7), (12, 13), (12, 9), (13, 10), (11, 2), (9, 11), (11, 4), (11, 3), (10, 13), (5, 13), (9, 13), (11, 13), (12, 7)] 
# ownpos = [2, 1]
# dist  = np.array(coins[np.argmin(np.linalg.norm(np.array(coins)-np.array(ownpos),axis=1))])-np.array(ownpos)
# print(dist)
# nearest_coin_rel = np.array(coins[np.argmin(np.linalg.norm(np.array(coins)-np.array(ownpos),axis=1))])-np.array(ownpos)
# surround = field[ownpos[0]-1:ownpos[0]+2,ownpos[1]-1:ownpos[1]+2].flatten()
# features = np.concatenate((nearest_coin_rel,surround))[None,:]
# print(features)
# 
# 
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# def create_q_model():
#     # start with Sequential Model input
#     input_shape = (11) #mean coins 2,next coin 2, own pos 2,  surround 9
#     model = keras.Sequential()
#     model.add(layers.Input(shape=(1,input_shape)))
#     model.add(layers.Dense(22, activation='relu'))
#     model.add(layers.Dense(22, activation='relu'))
#     model.add(layers.Dense(4, activation='linear'))
#     model.summary()
#     return model
# X = np.vstack((np.arange(12)[None,:],np.arange(1,13)[None,:]))
# Y = np.vstack((np.arange(4),np.arange(1,5)))
# model = create_q_model()
# model.compile(optimizer='adam', loss='binary_crossentropy')
# model.fit(X,Y)
# # =============================================================================
# # test = np.arange(12).reshape(None,1,11)
# # model.predict(test, training=False)
# # 
# # =============================================================================
# 
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# 
# =============================================================================





