# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 22:14:42 2022

@author: danie
"""

import numpy as np
import tensorflow as tf


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

def setup(self):
    """
    minimum setup
    """
    self.logger.info("Loading model.")
    self.model = tf.keras.models.load_model('model')
    self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),loss=tf.keras.losses.MeanSquaredError())

# python main.py play --agents ac9x9_superior_gen2 --train 0 --scenario classic --no-gui --n-rounds 10

def act(self, game_state):
    """
    minimum act
    """
    self.logger.info('Predicting Action')
    features0,features1,features2,features3 = state_to_features(game_state)
        
    feature_tensor0 = tf.expand_dims(features0, 0)
    action_probs0 = self.model(feature_tensor0,training=False)[0]
    action_probs0 = action_probs0/sum(abs(action_probs0))
    
    feature_tensor1 = tf.expand_dims(features1, 0)
    action_probs1 = perm(self.model(feature_tensor1,training=False)[0])
    action_probs1 = action_probs1/sum(abs(action_probs1))
    
    feature_tensor2 = tf.expand_dims(features2, 0)
    action_probs2 = perm(perm(self.model(feature_tensor2,training=False)[0]))
    action_probs2 = action_probs2/sum(abs(action_probs2))
    
    feature_tensor3 = tf.expand_dims(features3, 0)
    action_probs3 = perm(perm(perm(self.model(feature_tensor3,training=False)[0])))
    action_probs3 = action_probs3/sum(abs(action_probs3))

    action_probs = np.sum(np.vstack((action_probs0,action_probs1,action_probs2,action_probs3)),axis=0)
    
    self.logger.debug(f"Using argmax policy {action_probs}")
    
    action = np.argmax(action_probs)
    
    self.logger.debug(f"Choosing action {ACTIONS[action]}.")
    return ACTIONS[action]

k1 = {
      0:1,
      1:2,
      2:3,
      3:0      
      }
def perm(actions): # 0 1 2 3 to 3 0 1 2 
    act45 = actions[-2:]
    newact = np.zeros(4)
    for k in range(4): 
        newact[k] = actions[k1.get(k)]
    return np.concatenate((newact,act45))

def state_to_features(game_state: dict) -> np.array:
    """
    minimum features
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return [None]
    field = game_state['field']
    exp_map = game_state['explosion_map']
    ownpos = list(game_state['self'][3])
    
    n=3
    
    expand_f = np.zeros(tuple(s+6 for s in field.shape))    
    expand_f[tuple(slice(n,-n) for s in field.shape)] = field
    f_ac = expand_f[ownpos[0]+n-4:ownpos[0]+n+5,ownpos[1]+n-4:ownpos[1]+n+5]

    expand_em = np.zeros(tuple(s+6 for s in field.shape))    
    expand_em[tuple(slice(n,-n) for s in field.shape)] = exp_map
    em_ac = expand_em[ownpos[0]+n-4:ownpos[0]+n+5,ownpos[1]+n-4:ownpos[1]+n+5]
    
    expand_cab = np.zeros(tuple(s+6 for s in field.shape)) 
    coinsandbombs = np.zeros(field.shape)
    for coin in game_state['coins']:
        coinsandbombs.itemset(coin,1)
    for bomb in game_state['bombs']:
        coinsandbombs.itemset(bomb[0],-1/(1+bomb[1]))

    expand_cab[tuple(slice(n,-n) for s in field.shape)] = coinsandbombs
    cab_ac = expand_cab[ownpos[0]+n-4:ownpos[0]+n+5,ownpos[1]+n-4:ownpos[1]+n+5]
    
    expand_sap = np.zeros(tuple(s+6 for s in field.shape)) 
    selfandplayers = np.zeros(field.shape)
    if game_state['self'][2]:
        selfandplayers.itemset(game_state['self'][3],1)
        
    for player in game_state['others']:
        selfandplayers.itemset(player[3],-1)     
        
    expand_sap[tuple(slice(n,-n) for s in field.shape)] = selfandplayers
    sap_ac = expand_sap[ownpos[0]+n-4:ownpos[0]+n+5,ownpos[1]+n-4:ownpos[1]+n+5]
        
    return np.stack((f_ac,em_ac,cab_ac,sap_ac),axis=-1),np.stack((np.rot90(f_ac),np.rot90(em_ac),np.rot90(cab_ac),np.rot90(sap_ac)),axis=-1),np.stack((np.rot90(f_ac,k=2),np.rot90(em_ac,k=2),np.rot90(cab_ac,k=2),np.rot90(sap_ac,k=3)),axis=-1),np.stack((np.rot90(f_ac,k=3),np.rot90(em_ac,k=3),np.rot90(cab_ac,k=3),np.rot90(sap_ac,k=3)),axis=-1)

