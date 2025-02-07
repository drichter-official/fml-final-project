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
TRANSITION_HISTORY_SIZE = 1000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
num_actions = len(ACTIONS)
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
    self.logger.info(f"GPUs Available: {tf.config.list_physical_devices('GPU')}")
    if os.path.exists('model'):
        self.logger.info("Retraining from saved state.")
        self.model = tf.keras.models.load_model('model', compile=False)
        self.model_ref = tf.keras.models.load_model('model', compile=False)
    else:
        self.model = create_q_model()
        self.model_ref = create_q_model()
        
    self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='mse')
    self.model_ref.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='mse')
    self.total_rewards = []
    self.episode_length = []
    self.counter = 0
    
    self.gamma = 0.9
# python main.py play --agents task1_alt --train 1 --no-gui  --scenario coin-heaven  --n-rounds 100
num_actions = len(ACTIONS)
def create_q_model():
    # start with Sequential Model input
    input_shape = (4) #mean coins 2,next coin 2, own pos 2,  surround 9
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(num_actions, activation='linear'))
    model.summary()
    return model
    
# =============================================================================
#     input_shape = (17*17*2+2+1) #game state, coin  field, self pos, game step
#     model = keras.Sequential()
#     model.add(layers.Input(shape=(1,input_shape)))
#     model.add(layers.Conv1D(filters = 256,kernel_size=5, padding='same', activation='relu'))
#     model.add(layers.Conv1D(filters = 128,kernel_size=10, padding='same', activation='relu'))
#     model.add(layers.Conv1D(filters = 64,kernel_size=20, padding='same', activation='relu'))
#     model.add(layers.Dense(20, activation='relu'))
#     model.add(layers.Dense(num_actions, activation='linear'))
#     model.summary()
#     return model
#     
# =============================================================================
    
    # Network defined by the Deepmind paper
# =============================================================================
#     input_shape = (3,17,17)
#     cnn1 = keras.Sequential([
#         layers.Conv2D(16, kernel_size=(2,2), activation='relu',input_shape=input_shape),
#         layers.MaxPooling2D(pool_size=(2, 2),strides=2),
#         layers.Conv2D(32, kernel_size=(1, 1), activation='relu',input_shape=input_shape),
#         layers.MaxPooling2D(pool_size=(1, 1),strides=2),
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(num_actions, activation='linear')
#     ])
#     return cnn1
# # =============================================================================
# =============================================================================
#     inputs = layers.Input(shape=(17,17,3))#entire game, coin_field , self score x and y
# 
#     # Convolutions 
#     layer1 = layers.Conv2D(17, 6,activation="relu")(inputs)
#     layer2 = layers.Conv2D(10, 8,activation="relu")(layer1)
#     layer3 = layers.Conv2D(5, 10,activation="relu")(layer2)
# 
#     layer4 = layers.Flatten()(layer3)
# 
#     layer5 = layers.Dense(64, activation="relu")(layer4)
#     action = layers.Dense(num_actions, activation="linear")(layer5)
# 
# 
#     return keras.Model(inputs=inputs, outputs=action)
# =============================================================================

# =============================================================================
# epsilon_min = 0.1  # Minimum epsilon greedy parameter
# epsilon_max = 1.0  # Maximum epsilon greedy parameter
# epsilon_interval = (
#     epsilon_max - epsilon_min
# )  # Rate at which to reduce chance of random action being taken
# =============================================================================
# Size of batch taken from replay buffer
batch_size = 32   
# Train the model after 5 actions
update_after_actions = 5

loss_function = keras.losses.MeanSquaredError() #  reduction=tf.keras.losses.Reduction.SUM
update_target_network = 100*update_after_actions




def train(self):
    
    batch = random.sample(self.transitions,batch_size)
    
    old_state_list = np.array([transition[0]  if not None in transition[2] else transition[0] for transition in batch])#tf.experimental.numpy.vstack([tf.expand_dims(transition[0],axis=0) for transition in batch])
    old_qs_list = self.model.predict(old_state_list,batch_size=batch_size)
    
    new_state_list = np.array([transition[2]  if not None in transition[2] else np.array([0,0,0,0]) for transition in batch])
    new_qs_list = self.model_ref.predict(new_state_list,batch_size=batch_size)
    
    x = []
    y = []
    for index, (old_state, action, new_state, reward) in enumerate(batch):
        if None in new_state:   # if this transition is from end_of_round
            new_q = reward
        else:
            max_new_q = np.max(new_qs_list[index])
            new_q = reward + self.gamma * max_new_q

        old_qs = old_qs_list[index]
        old_qs[action] = new_q

        x.append(old_state)
        y.append(old_qs)

    self.model.fit(np.array(x), np.array(y), batch_size=batch_size,verbose=0, shuffle=False)

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
    if None in state_to_features(old_game_state):
        self.logger.debug("Old State None Error")
        pass
    else:
        self.transitions.append(
            Transition(
                state_to_features(old_game_state),
                ACTIONStonum.get(self_action),
                state_to_features(new_game_state),
                reward_from_events(self, events)
                )
            )
        self.counter += 1
        if self.counter % update_after_actions == 0 and len(self.transitions) > batch_size+2: 
            self.logger.debug(f"Training model...")
            train(self)
            if self.counter % update_target_network == 0:
                self.model_ref.set_weights(self.model.get_weights())

            
            
# =============================================================================
#         indices = np.random.choice(np.arange(1,transit_count-2)+1, size=batch_size)         
# =============================================================================
 
# =============================================================================
#         state_sample = tf.experimental.numpy.vstack([self.transitions[i][0] for i in indices if self.transitions[i][2]!=None  and self.transitions[i][0]!=None ]) # list of tensors
# # =============================================================================
# #             state_sample = tf.data.Dataset.from_tensor_slices(state_sample).batch(1)
# # =============================================================================
#         action_sample = np.array([self.transitions[i][1] for i in indices if self.transitions[i][2]!=None and self.transitions[i][0]!=None])# list of actions
#         state_next_sample = [self.transitions[i][2] for i in indices if self.transitions[i][2]!=None and self.transitions[i][0]!=None ]
#         state_next_sample = tf.experimental.numpy.vstack([tf.expand_dims(self.transitions[i][2],axis=0) for i in indices if self.transitions[i][2]!=None and self.transitions[i][0]!=None ])
#         rewards_sample = np.array([self.transitions[i][3] for i in indices if self.transitions[i][2]!=None and self.transitions[i][0]!=None ],dtype=np.float32)
#         
# # =============================================================================
# #             self.logger.info(f'state_next_sample {state_next_sample}')
# # =============================================================================
#         
#         future_rewards = self.model_ref.predict(state_next_sample)
# # =============================================================================
# #         self.logger.info(f'tf.expand_dims(tf.convert_to_tensor(rewards_sample), axis=1) {tf.convert_to_tensor(rewards_sample),type(gamma * tf.reduce_max(future_rewards, axis=1))}')
# # =============================================================================
#         updated_q_values = tf.add(tf.convert_to_tensor(rewards_sample), gamma * tf.reduce_max(future_rewards, axis=1))
# # =============================================================================
# #             self.logger.info(f'updated_q_values,rewards_sample {updated_q_values,rewards_sample}')
# # =============================================================================
#         masks = tf.one_hot(action_sample, num_actions)
#         
#         with tf.GradientTape() as tape:
#             # Train the model on the states and updated Q-values
#             q_values = self.model(state_sample, training=False)
# 
#             # Apply the masks to the Q-values to get the Q-value for action taken
#             q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
# # =============================================================================
# #             self.logger.info(f'q_action{q_action}')
# # =============================================================================
#             # Calculate loss between new Q-value and old Q-value
#             loss = loss_function(updated_q_values, q_action)
# # =============================================================================
# #             self.logger.info(f'loss{loss}')
# # =============================================================================
#         # Backpropagation
#         grads = tape.gradient(loss, self.model.trainable_variables)
# # =============================================================================
# #         self.logger.info(f'self.model.trainable_variables{self.model.trainable_variables}')
# #         self.logger.info(f'grads{grads}')
# # =============================================================================
#         processed_grads = [tf.clip_by_value(g,clip_value_min=0, clip_value_max=1) for g in grads]
#         optimizer.apply_gradients(zip(processed_grads, self.model.trainable_variables))
#         # Get indices of samples for replay buffers
#         self.model.compile(optimizer=optimizer)
#         self.logger.info('Model updated')
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
            ACTIONStonum.get(last_action),
            [None],
            reward_from_events(self, events)
            )
        )

    # Store the model:
    # update the the target network with new weights
    
    tot_reward = 0
    for transit in self.transitions:
        tot_reward  += transit[3]
    self.total_rewards.append(tot_reward)
    self.episode_length.append(last_game_state['step'])
    with open("rewards.pt", "wb") as file:
        pickle.dump(self.total_rewards , file)
    with open("episodelength.pt", "wb") as file:
        pickle.dump(self.episode_length , file)
        
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    
    tf.keras.models.save_model(self.model_ref,'model')

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






