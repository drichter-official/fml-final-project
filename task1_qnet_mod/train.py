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
TRANSITION_HISTORY_SIZE = 399 # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
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
        self.model = tf.keras.models.load_model('model', compile=False)
        self.target_model = tf.keras.models.load_model('model', compile=False)
    else:
        self.model = create_q_model()
        self.target_model = create_q_model()
        self.target_model.set_weights(self.model.get_weights())
    self.tot_rewards = []
# python main.py play --agents task1 --no-gui --n-rounds 100
num_actions = len(ACTIONS)
ACTIONSdict = {ACTIONS[i] : i for i in range(num_actions)}
def create_q_model():
    # start with Sequential Model input
    input_shape = (11) #mean coins 2,next coin 2, own pos 2,  surround 9
    model = keras.Sequential()
    model.add(layers.Input(shape=(1,input_shape)))
    model.add(layers.Dense(22, activation='relu'))
    model.add(layers.Dense(22, activation='relu'))
    model.add(layers.Dense(num_actions, activation='linear'))
    model.summary()
    return model
    

# # =============================================================================
# # epsilon_min = 0.1  # Minimum epsilon greedy parameter
# # epsilon_max = 1.0  # Maximum epsilon greedy parameter
# # epsilon_interval = (
# #     epsilon_max - epsilon_min
# # )  # Rate at which to reduce chance of random action being taken
# # =============================================================================
# # Size of batch taken from replay buffer
# batch_size = 16  
# # Train the model after 5 actions
# update_after_actions = 5
# loss_function = keras.losses.Huber()
# update_target_network = 100*update_after_actions
# =============================================================================
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
            self_action,
            state_to_features(new_game_state),
            reward_from_events(self, events),
            )
        )
    
# =============================================================================
#     if epsilon > np.random.rand(1)[0]:
#         # Take random action
#         action = np.random.choice(num_actions)
#     else:
#         coin_field  = np.zeros(shape=self.transition[-1].state["field"].shape)
#         coins = self.transition[-1].state["coins"]
#         for coin in coins:
#             coin_field.itemset(coin,1)
#         state = tf.convert_to_tensor([self.transition[-1].state["field"].flatten(),coin_field.flatten()])
#         state_tensor = tf.convert_to_tensor(state)
#         state_tensor = tf.expand_dims(state_tensor, 0)
#         action_probs = self.model(state_tensor, training=False)
#     epsilon = epsilon_decay*epsilon
#     action = tf.argmax(action_probs[0]).numpy()
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
    self.model.compile(optimizer='adam', loss='binary_crossentropy')
    self.target_model.compile(optimizer='adam', loss='binary_crossentropy')
# =============================================================================
#     self.logger.debug(f'{self.transitions}')
#     self.logger.debug(f'{len(self.transitions)}')
# =============================================================================
    
    
    
    # Get a minibatch of random samples from memory replay table
    self.transitions.pop()
    batch = random.sample(self.transitions, 300)
# =============================================================================
#     self.logger.debug(f'batch{batch}')
# =============================================================================
    # Get current states from batch, then query NN model for Q value
    old_states = np.array([transit[0] for transit in batch])
    self.logger.debug(f'old_states{old_states}')
    old_qs_list = self.model.predict(old_states)
# =============================================================================
#     self.logger.debug(f'self.model.predict(old_states){self.model.predict(old_states)}')
# =============================================================================
    
    # Get future states from minibatch, then query NN model for Q values
    # When using target network, query it, otherwise main network should be queried
    new_states = np.array([transit[2] for transit in batch])
    self.logger.debug(f'new_states{new_states}')
# =============================================================================
#     new_state_tensors = tf.convert_to_tensor(new_states)
# =============================================================================
# =============================================================================
#     new_state_tensors = tf.expand_dims(new_state_tensors, 1) 
# =============================================================================
    future_qs_list = self.target_model.predict_on_batch(new_states)
    X = []
    y = []
    tot_reward = 0
    # Now we need to enumerate our batches
    for index, (old_state, action, new_state, reward) in enumerate(batch):
        tot_reward+=reward
        # If not a terminal state, get new q from future states, otherwise set it to 0
        # almost like with Q Learning, but we use just part of equation here
# =============================================================================
#         if new_state.all()!=None:
# =============================================================================
        max_future_q = np.max(future_qs_list[index])
        new_q = reward + 0.9 * max_future_q
# =============================================================================
#         else:
#             new_q = reward
# =============================================================================

        # Update Q value for given state
        old_qs = old_qs_list[index]
        self.logger.debug(f'old_qs_list[index]{old_qs_list[index]}')
        self.logger.debug(f'action{action}')
        old_qs[:,ACTIONSdict.get(action)] = new_q

        # And append to our training data
        X.append(old_state)
        y.append(old_qs)

    # Fit on all samples as one batch, log only on terminal state
    self.model.fit(np.array(X), np.array(y), batch_size=300, verbose=0, shuffle=False, callbacks = None)
       
    
    

    # Store the model:
    # update the the target network with new weights
    self.target_model.set_weights(self.model.get_weights())
    self.target_model.compile(optimizer='adam', loss='binary_crossentropy')
    # Log details
    self.logger.debug(
        f'weights are {self.model.get_weights()}'
    )
    self.tot_rewards.append(tot_reward)
    with open("rewards.pt", "wb") as file:
        pickle.dump(self.tot_rewards, file)
    tf.keras.models.save_model(self.target_model,'model')

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        # e.KILLED_OPPONENT: 5,
        # e.PLACEHOLDER_EVENT: -0.1,  # idea: the custom event is bad
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






