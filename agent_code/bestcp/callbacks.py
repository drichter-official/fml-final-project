# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:39:46 2022

@author: danie
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 18:15:14 2022

@author: danie
"""
import os
import numpy as np
import tensorflow as tf

from autoanalysis import k

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.debug('Successfully entered setup code')

    self.logger.info("Loading model.")
    self.model = tf.keras.models.load_model('model')
    self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),loss=tf.keras.losses.Huber())
    path = "C:/Users/danie/Meine Ablage/Github/bomberman_rl/agent_code/bestcp/Checkpoints/"
    #self.cpts = list(set([a+"."+b for a,b in [f.split(".")[0:2] for f in os.listdir(path)]]))
    
    #print(self.cpts)
    #print(f"loading {['ckpt-1-rew_56.6','checkpoint-eps_4400-rew_68.9-1','checkpoint-eps_5600-rew_69.5-1', 'checkpoint-eps_600-rew_48.3-1', 'checkpoint-eps_9800-rew_48.0-1', 'ckpt-2-rew_62.4', 'checkpoint-eps_4800-rew_51.9-1', 'checkpoint-eps_12200-rew_53.7-1', 'checkpoint-eps_5000-rew_59.0-1'][k]}")
    #self.model.load_weights(path+'checkpoint-eps_5600-rew_69.5-1')
    tf.keras.models.save_model(self.model,'model')

def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
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
    self.logger.debug(f"Using argmax policy{action_probs0,action_probs1,action_probs2,action_probs3}")
    self.logger.debug(f"Using argmax policy{action_probs}")

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
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
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

