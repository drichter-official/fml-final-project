# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 11:13:26 2022

@author: ztec1
"""

import pickle
import random
from collections import namedtuple, deque
from typing import List
import os
import events as e

from .callbacks import state_to_features
from .callbacks import ACTIONS
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



#%%
# This is only an example!
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10 # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
num_actions = len(ACTIONS)
num_states = 6
ACTIONStonum = {ACTIONS[i] : i for i in range(num_actions)}
def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    
    if os.path.exists('model.pt'):
        self.logger.info("Retraining from saved state.")
        self.logger.info("Loading model.")
        with open("model.pt", "rb") as file:
            self.Q_table = pickle.load(file)
    else:
        self.Q_table =  np.zeros((14,14,2,2, num_actions))
    self.alpha = 0.1
    self.gamma = 0.7

    self.total_rewards = []
# python main.py play --agents task1 --train 1 --no-gui  --scenario coin-heaven --n-rounds 100
# python main.py play --agents task1 --train 0  --scenario coin-heaven --n-rounds 1 

def create_q_model():
    # start with Sequential Model input
    input_shape = (6) #mean coins 2,next coin 2, own pos 2,  surround 9
    model = keras.Sequential()
    model.add(layers.Input(shape=(1,input_shape)))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(12, activation='relu'))
    model.add(layers.Dense(num_actions, activation='linear'))
    model.summary()
    return model

def game_events_occurred(
        self,
        old_game_state: dict,
        self_action: str,
        new_game_state: dict,
        events: List[str],
        ):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}'
    )
    self.transitions.append(
        Transition(
            state_to_features(old_game_state),
            ACTIONStonum.get(self_action),
            state_to_features(new_game_state),
            reward_from_events(self, events),
            )
        )
    last_state,action,new_state,reward = self.transitions[-1]
    if None in new_state or None in last_state: #if  not finished
        pass
    else: # update q-table
    # calculate q value for  given state and  action
# =============================================================================
#         self.logger.debug(f'{last_state}')
# =============================================================================
        q_val = self.Q_table[last_state[0],last_state[1],last_state[2],last_state[3],action]
        max_val = np.max(self.Q_table[new_state[0],new_state[1],new_state[2],new_state[3],:])
        new_q_val = (1 - self.alpha) * q_val + self.alpha * (reward + self.gamma * max_val)
# =============================================================================
#         self.logger.debug(
#             f' new_q_val{ new_q_val}'
#         )
# =============================================================================

        self.Q_table[last_state[0],last_state[1],last_state[2],last_state[3], action] = new_q_val

def end_of_round(
        self,
        last_game_state: dict,
        last_action: str,
        events: List[str]
        ): 
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(
        Transition(
            state_to_features(last_game_state),
            last_action,
            None,
            reward_from_events(self, events)
            )
        )
    # Store the model
    with open(r"model.pt", "wb") as file:
        pickle.dump(self.Q_table, file)
    # Calculate total reward for analysis
    tot_reward = 0
    for transit in self.transitions:
        tot_reward  += transit[3]
    self.total_rewards.append(tot_reward)
    with open("rewards.pt", "wb") as file:
        pickle.dump(self.total_rewards , file)
# # =============================================================================
# #     self.logger.debug(f'{self.transitions}')
# #     self.logger.debug(f'{len(self.transitions)}')
# # =============================================================================
#     
#     
#     
#     # Get a minibatch of random samples from memory replay table
#     self.transitions.pop()
#     batch = random.sample(self.transitions, 300)
# # =============================================================================
# #     self.logger.debug(f'batch{batch}')
# # =============================================================================
#     # Get current states from batch, then query NN model for Q value
#     old_states = np.array([transit[0] for transit in batch])
#     self.logger.debug(f'old_states{old_states}')
#     old_qs_list = self.model.predict(old_states)
# # =============================================================================
# #     self.logger.debug(f'self.model.predict(old_states){self.model.predict(old_states)}')
# # =============================================================================
#     
#     # Get future states from minibatch, then query NN model for Q values
#     # When using target network, query it, otherwise main network should be queried
#     new_states = np.array([transit[2] for transit in batch])
#     self.logger.debug(f'new_states{new_states}')
# # =============================================================================
# #     new_state_tensors = tf.convert_to_tensor(new_states)
# # =============================================================================
# # =============================================================================
# #     new_state_tensors = tf.expand_dims(new_state_tensors, 1) 
# # =============================================================================
#     future_qs_list = self.target_model.predict_on_batch(new_states)
#     X = []
#     y = []
#     tot_reward = 0
#     # Now we need to enumerate our batches
#     for index, (old_state, action, new_state, reward) in enumerate(batch):
#         tot_reward+=reward
#         # If not a terminal state, get new q from future states, otherwise set it to 0
#         # almost like with Q Learning, but we use just part of equation here
# # =============================================================================
# #         if new_state.all()!=None:
# # =============================================================================
#         max_future_q = np.max(future_qs_list[index])
#         new_q = reward + 0.9 * max_future_q
# # =============================================================================
# #         else:
# #             new_q = reward
# # =============================================================================
# 
#         # Update Q value for given state
#         old_qs = old_qs_list[index]
#         self.logger.debug(f'old_qs_list[index]{old_qs_list[index]}')
#         self.logger.debug(f'action{action}')
#         old_qs[:,ACTIONSdict.get(action)] = new_q
# 
#         # And append to our training data
#         X.append(old_state)
#         y.append(old_qs)
# 
#     # Fit on all samples as one batch, log only on terminal state
#     self.model.fit(np.array(X), np.array(y), batch_size=300, verbose=0, shuffle=False, callbacks = None)
#        
#     
#     
# 
#     # Store the model:
#     # update the the target network with new weights
#     self.target_model.set_weights(self.model.get_weights())
#     self.target_model.compile(optimizer='adam', loss='binary_crossentropy')
#     # Log details
#     self.logger.debug(
#         f'weights are {self.model.get_weights()}'
#     )
#     self.tot_rewards.append(tot_reward)
# =============================================================================

# =============================================================================
#     tf.keras.models.save_model(self.target_model,'model')
# =============================================================================

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        # e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -1,
        e.WAITED: -0.1,
        # e.KILLED_SELF: -5,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum






