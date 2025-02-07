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
    
    if self.train or not os.path.isdir("model"):
        self.logger.info("Setting up new model.")

    else:
        self.logger.info("Loading model.")
        self.model = tf.keras.models.load_model('model')
#%%

#python main.py play --agents task_1 --train 1 --scenario coin-heaven
def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Predicting Action')
    features = state_to_features(game_state)
    epsilon = 0.4
    if self.train and epsilon > np.random.rand(1)[0]:
        self.logger.debug("Epsilon-greedy: Choosing action purely at random.")
        # Take random action
        action = int(np.random.choice(len(ACTIONS),1))
    else:
# =============================================================================
#         action = np.argmax(self.model.predict(features)[0])
# =============================================================================
        action_probs = self.model(tf.expand_dims(features, 0), training=False)
        action = tf.argmax(action_probs[0,0]).numpy()
# =============================================================================
#         self.logger.debug(f"action_probs{action_probs}.")
# =============================================================================
        
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
        return None
    coins=game_state['coins']
    coin_field  = np.zeros(shape=game_state["field"].shape)
    ownpos = list(game_state['self'][3])
    nearest_coin_rel = np.array(coins[np.argmin(np.linalg.norm(np.array(coins)-np.array(ownpos),axis=1))])-np.array(ownpos)
    for coin in game_state["coins"]:
        coin_field.itemset(coin,1)
    gameandcoins = np.array(game_state['field']+coin_field)
    surround = gameandcoins[ownpos[0]-1:ownpos[0]+2,ownpos[1]-1:ownpos[1]+2].flatten()
    features = np.concatenate((nearest_coin_rel,surround))[None,:]
    state_tensor = tf.convert_to_tensor(list(features))
    return state_tensor
# =============================================================================
#     coin_field  = np.zeros(shape=game_state["field"].shape)
#     ownpos = np.zeros(shape=game_state["field"].shape)
#     for coin in game_state["coins"]:
#         coin_field.itemset(coin,1)
#     ownpos.itemset(game_state['self'][3],1)
#     features = [game_state["field"],coin_field,ownpos]
#     
#     return features
# # =============================================================================
# =============================================================================
#     features = np.concatenate((
#             np.array([game_state['round']]),
#             np.array([game_state['step']]),
#             game_state['field'],
#             np.array(game_state['coins']),
#             np.array(game_state['self'])
#              )
#         )
#     print(features)
# =============================================================================
    # For example, you could construct several channels of equal shape, ...
# =============================================================================
#     if game_state["coins"] != []:
#         distance = game_state["coins"] - np.array(game_state["self"][3])
#         closest_index = np.argmin(np.sum(np.abs(distance), axis=1))
#         closest_vector = distance[closest_index]
#     else:
#         closest_vector = [0, 0]  # treat non-existing coins as [0,0]
# 
# =============================================================================
    # check if surrounding tiles are blocked
    # x_off = [1, -1, 0, 0]
    # y_off = [0, 0, 1, -1]
    # blocked = np.zeros(4)
    # for i in range(len(blocked)):
    #    blocked[i] = game_state["field"][
    #        game_state["self"][3][0] + x_off[i], game_state["self"][3][1] + y_off[i]
    #    ]

# =============================================================================
#     mod_pos = [game_state["self"][3][0] % 2, game_state["self"][3][1] % 2]
# 
# =============================================================================
    # channels = []
    # channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    # stacked_channels = np.stack(channels)
    # and return them as a vector
    return features  # stacked_channels.reshape(-1)