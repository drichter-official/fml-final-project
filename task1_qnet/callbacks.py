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
    self.epsilon = 1# start with 100%  exploration
    self.epsilon_min = 0.1 # while training explore at  least  10% of steps
    self.epsilon_decay = 0.99
    
    self.epsilon_imitate = 0.80 # percentage of random moves get imitatet
    self.epsilon_imitate_decay = 0.99
    
    self.softmax = True
    self.imitate = False
    
    if self.train or not os.path.isdir("model"):
        self.logger.info("Setting up new model.")

    else:
        self.logger.info("Loading model.")
        self.model = tf.keras.models.load_model('model')
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='mse')
        

#python main.py play --agents task1_alt --train 0 --scenario coin-heaven
def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    if game_state['round'] != self.episode: # keep track with episodes
        # keep track of episode
        self.logger.info(f"New Episode{game_state['round']}")
        self.episode +=1
        # and update epsilon after each episode
        self.logger.info(f"self.epsilon{self.epsilon}")
        self.epsilon = self.epsilon*self.epsilon_decay
        
        
    self.logger.info('Predicting Action')
    features = state_to_features(game_state)
    feature_tensor = tf.expand_dims(features, 0)
    rand = np.random.rand(1)[0]
    action = 0
    
    if (self.epsilon >  rand or self.epsilon_min  > rand) and self.train:
        if not self.softmax:
            self.logger.debug("Epsilon-greedy: Choosing action purely at random.")
            action = int(np.random.choice(len(ACTIONS),1))
        else:
            action_probs = self.model(feature_tensor)
            probs = tf.keras.activations.softmax(action_probs,axis=-1)[0]
            self.logger.debug(f"Using Softmax policy: Choosing action with probability{probs}")
            action = np.dot(np.arange(len(ACTIONS)),np.random.multinomial(1,probs))
        self.logger.debug(f"Choosing action {ACTIONS[action]}.")
    else:
        action_probs = self.model(feature_tensor)
        self.logger.debug(f"Using argmax policy")
        action = np.argmax(action_probs[0])     
        
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
        return [None]
    
    coins=game_state['coins']
    if coins != []:      
        ownpos = list(game_state['self'][3])
        nearest_coin_rel = np.array(coins[np.argmin(np.linalg.norm(np.array(coins)-np.array(ownpos),axis=1))])-np.array(ownpos)
        if np.linalg.norm(nearest_coin_rel) != 0:
            nearest_coin_norm = nearest_coin_rel/np.linalg.norm(nearest_coin_rel)
        else:
            nearest_coin_norm =[0,0]
    else:
        nearest_coin_norm =[0,0]

    ownpos = list(game_state['self'][3])

    surround = game_state['field'][ownpos[0]-1:ownpos[0]+2,ownpos[1]-1:ownpos[1]+2].flatten()
    mod_pos = [game_state["self"][3][0] % 2, game_state["self"][3][1] % 2]
    features = np.concatenate((nearest_coin_norm,mod_pos))
    return tf.convert_to_tensor(features)
