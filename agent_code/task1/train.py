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




# This is only an example!
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 100 # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

num_actions = len(ACTIONS)
num_states = 14*14*2*2

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
    
    if os.path.exists('model'):
        self.logger.info("Retraining from saved state.")
        self.model = tf.keras.models.load_model('model', compile=True)
        self.model_target = tf.keras.models.load_model('model', compile=True)
    else:
        self.model = create_q_model()
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.model_target = create_q_model()
        self.model_target.set_weights(self.model.get_weights())
        self.model_target.compile(optimizer='adam', loss='binary_crossentropy')
# =============================================================================
#     self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
#     self.loss_function = keras.losses.Huber()
# =============================================================================
    self.model.summary()
    # PARAMETERS
    self.alpha = 0.2
    self.gamma = 0.8
    self.batch = 16 # use ... random events from TransitHistory
    self.update_intervall = 5
    self.total_rewards = []
# python main.py play --agents task1 --train 1 --no-gui  --scenario coin-heaven --n-rounds 100
# python main.py play --agents task1 --train 0  --scenario coin-heaven --n-rounds 1 

def create_q_model():
    # start with Sequential Model input
    input_shape = (4) #next coin 2 and mod
    model = keras.Sequential()
    model.add(layers.Input(shape=(1,input_shape)))
    model.add(layers.Dense(12, activation='relu'))
    model.add(layers.Dense(12, activation='relu'))
    model.add(layers.Dense(num_actions, activation='linear'))
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
# =============================================================================
#     self.logger.info(f"new_game_state{new_game_state}")
# =============================================================================
    
    if new_game_state['step'] % self.update_intervall  == 0 and (len(self.transitions))-5>self.batch:
        # update every  ... steps and begin  when enough data  is  collected
        self.transitions.popleft()
        self.transitions.pop()
        batch = random.sample(self.transitions, self.batch)
        old_states = np.array([transit[0][None,:] for transit in batch])
        old_q_list = self.model.predict(old_states)
        new_states = np.array([transit[2][None,:] for transit in batch])
        new_q_list = self.model_target.predict(new_states)
        X = []
        y = []
        for ind, (old_state,action,new_state,reward) in enumerate(batch):
            if None in old_state or None in new_state:
                new_q_val = reward 
            else:
                max_future_q = np.max(new_q_list[ind])
                new_q_val = reward + self.gamma * max_future_q
            old_q = old_q_list[ind]
            old_q[:,action] = new_q_val 
            X.append(old_state[None,:])
            y.append(list(old_q))
    
        self.model.fit(np.array(X), np.array(y).squeeze(axis=1), batch_size=self.batch, verbose=0, shuffle=False)
        self.logger.debug('Modelrefitted')
# =============================================================================
#     if new_game_state['step'] % self.update_intervall  == 0 and (len(self.transitions))-5>self.batch:
#         # update every  ... steps and begin  when enough data  is  collected
#         self.transitions.popleft()
#         self.transitions.pop()
#         batch = random.sample(self.transitions, self.batch)
#         old_states = np.array([transit[0][None,:] for transit in batch])
#     
#         old_q_list = self.model.predict(old_states)
#         self.logger.info(f"old_q_list{old_q_list}")
#         new_states = np.array([transit[2][None,:] for transit in batch])
#         new_q_list = self.model_target.predict(new_states)
#         self.logger.info(f"new_q_list{new_q_list}")
#         X = []
#         y = []
#         for ind, (old_state,action,new_state,reward) in enumerate(batch):
#             if None in old_state or None in new_state:
#                 new_q_val = reward 
#             else:
#                 max_future_q = np.max(new_q_list[ind])
#                 self.logger.info(f"new_q_list[ind]{new_q_list[ind]}")
#                 new_q_val = reward + self.gamma * max_future_q
#                 
#             old_q = old_q_list[ind]
#             self.logger.info(f"old_q{old_q_list[ind]}")
#             old_q[:,action] = new_q_val 
#             self.logger.info(f"old_q{type(old_q)}")
#             X.append(old_state[None,:])
#             y.append(list(old_q))
#         
#         self.logger.info(f"X{np.array(X).shape}")
#         self.logger.info(f"y{np.array(y).shape}")
#         self.logger.info(f"X{np.array(X)}")
#         self.logger.info(f"y{np.array(y)}")
#         self.model.fit(np.array(X), np.array(y).squeeze(axis=1), batch_size=self.batch, verbose=0, shuffle=False)
#         self.logger.debug('Modelrefitted')
# =============================================================================

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
    # Update target network
    self.model_target.set_weights(self.model.get_weights())
    # Save the model
    tf.keras.models.save_model(self.model_target,'model')
    # Calculate total reward for analysis
    tot_reward = 0
    for transit in self.transitions:
        tot_reward  += transit[3]
    self.total_rewards.append(tot_reward)
    with open("rewards.pt", "wb") as file:
        pickle.dump(self.total_rewards , file)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

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






