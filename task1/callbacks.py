# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 09:41:16 2022

@author: ztec1
"""

import os
import pickle
import random

import numpy as np
import settings as s
import tensorflow as tf

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT"]  # , "BOMB"]


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
    self.episode = 1
    self.epsilon = 0.5  #  start with random exploration
    self.epsilon_min = 0.1 
    self.epsilon_decay = 0.99 #  decay by 1% every episode => 1000 epsiodes = approx 36% random
    if self.train or not os.path.exists("model.pt"):
        self.logger.info("Setting up new model.")

    else:
        self.logger.info("Loading existing model.")
        self.model_target = tf.keras.models.load_model('model', compile=True)

#python main.py play --agents task1 --train 1 --scenario coin-heaven
def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Predicting Action')
    state = state_to_features(game_state)
    action = 0 #initilize
    rand = np.random.rand(1)[0]
    if self.train:
        if game_state['round'] != self.episode: # keep track with episodes
            # keep track of episode
            self.logger.info(f"New Episode{game_state['round']}")
            self.episode +=1
            # and update epsilon after each episode
            self.logger.info(f"self.epsilon{self.epsilon}")
            self.epsilon = self.epsilon*self.epsilon_decay
            
        if  self.epsilon >  rand or self.epsilon_min  > rand :
            self.logger.debug("Epsilon-greedy: Choosing action purely at random.")
            # Take random action
            action = int(np.random.choice(len(ACTIONS),1))
    else:
        
        action = np.argmax(self.model_target.predict(state)[0])
        
    self.logger.debug(f"Choosing action {ACTIONS[action]}.")
    return ACTIONS[action]

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
        return 0
    coins=game_state['coins']
    if coins != []:      
        ownpos = list(game_state['self'][3])
        nearest_coin_rel = np.array(coins[np.argmin(np.linalg.norm(np.array(coins)-np.array(ownpos),axis=1))])-np.array(ownpos)
    else:
        nearest_coin_rel = [0, 0]  # treat non-existing coins as [0,0]
        
    mod_pos = [game_state["self"][3][0] % 2, game_state["self"][3][1] % 2]
    state = np.concatenate((nearest_coin_rel, mod_pos))
    state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
    return state
# =============================================================================
#     coins=game_state['coins']
#     coin_field  = np.zeros(shape=game_state["field"].shape)
#     ownpos = list(game_state['self'][3])
#     nearest_coin_rel = np.array(coins[np.argmin(np.linalg.norm(np.array(coins)-np.array(ownpos),axis=1))])-np.array(ownpos)
#     for coin in game_state["coins"]:
#         coin_field.itemset(coin,1)
#     gameandcoins = np.array(game_state['field']+coin_field)
#     surround = gameandcoins[ownpos[0]-1:ownpos[0]+2,ownpos[1]-1:ownpos[1]+2].flatten()
#     features = np.concatenate((nearest_coin_rel,surround))[None,:]
#     state_tensor = tf.convert_to_tensor(list(features))
#     return state_tensor
# =============================================================================
