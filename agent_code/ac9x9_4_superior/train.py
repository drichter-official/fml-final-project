# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 11:13:26 2022

@author: ztec1
"""
# python main.py play --agents 9x9ac_2 rule_based_agent rule_based_agent chicken --train 1 --scenario classic --n-rounds 4000 --no-gui
import pickle
import random
from collections import namedtuple, deque
from typing import List
import os
import events as e

from .callbacks import state_to_features
from .callbacks import ACTIONS
import agent_code.ac9x9_4_superior.Setting as S 

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

path = 0
# This is only an example!
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward","done"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = S.TRANSITHIST  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
LOOP = "LOOP"
KILLED_BY_OPPNENT = "KILLED_BY_OPPNENT"

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
    if S.LOADCHECKPOINT and os.path.exists('model'):
        self.model = create_q_model()
        self.model_ref = create_q_model()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=S.LEARNINGRATE, clipnorm=1.0),loss=tf.keras.losses.Huber())
        self.model_ref.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=S.LEARNINGRATE, clipnorm=1.0),loss=tf.keras.losses.Huber())
        #self.model.load_weights("C:/Users/danie/Meine Ablage/Github/bomberman_rl/agent_code/ac9x9_1_superior/BestCheck/checkpoint-eps_4800-rew_51.9-1")
        #self.model_ref.load_weights("C:/Users/danie/Meine Ablage/Github/bomberman_rl/agent_code/ac9x9_1_superior/BestCheck/checkpoint-eps_4800-rew_51.9-1")
        
        self.model.load_weights("C:/Users/danie/Meine Ablage/Github/bomberman_rl/agent_code/ac9x9_4_superior/BestCheck/checkpoint-eps_5000-rew_59.0-1")
        self.model_ref.load_weights("C:/Users/danie/Meine Ablage/Github/bomberman_rl/agent_code/ac9x9_4_superior/BestCheck/checkpoint-eps_5000-rew_59.0-1")
        #self.model.load_weights("C:/Users/danie/Meine Ablage/Github/bomberman_rl/agent_code/ac9x9_3_superior/BestCheck/checkpoint-eps_12200-rew_53.7-1")
        #self.model_ref.load_weights("C:/Users/danie/Meine Ablage/Github/bomberman_rl/agent_code/ac9x9_3_superior/BestCheck/checkpoint-eps_12200-rew_53.7-1")

        
    elif os.path.exists('model'):
        self.logger.info("Retraining from saved state.")
        self.model = tf.keras.models.load_model('model', compile=False)
        self.model_ref = tf.keras.models.load_model('model', compile=False)
        self.model.summary()
        self.model_ref.summary()
    else:
        self.model = create_q_model()
        self.model_ref = create_q_model()
    
    self.optimizer = keras.optimizers.Adam(learning_rate=S.LEARNINGRATE, clipnorm=1.0)
    self.loss_function = keras.losses.Huber()
        
    self.model.compile(optimizer=self.optimizer,loss=self.loss_function)
    self.model_ref.compile(optimizer=self.optimizer,loss=self.loss_function)
    
    self.checkpointcounter = 0
    self.counter = 0
    self.lastlastaction = None
    self.lastaction = None
    
    self.batch_size = S.BATCH_SIZE
    self.update_after_actions = S.UPDATE_ACTIONS
    self.update_target_network = S.UPDATE_TARGET
    
    self.gamma = S.GAMMA
    
    # Analytics
    self.total_rewards = []
    self.episodelength = []
    self.coins_collected = []
    
def create_q_model():
    input_shape = (9,9,4)
    model = keras.Sequential()
    model.add(layers.Conv2D(16, kernel_size=(5, 5), input_shape=input_shape))
    model.add(layers.Conv2D(32 , kernel_size=(3, 3), padding='valid',activation='relu', ))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='valid',activation='relu', ))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(num_actions, activation='linear'))
    model.summary()
    return model

def train(self):
    batch = random.sample(self.transitions,self.batch_size)
        
    old_state_list = np.array([transition[0] for transition in batch])#tf.experimental.numpy.vstack([tf.expand_dims(transition[0],axis=0) for transition in batch])
    action_list = np.array([transition[1] for transition in batch])
    new_state_list = np.array([transition[2] if not None in transition[2] else np.zeros((transition[0].shape)) for transition in batch])    
    rewards_list = np.array([transition[3] for transition in batch])
    done_list = tf.convert_to_tensor([float(transition[4]) for transition in batch])
    
    
    new_qs_list = self.model_ref.predict(new_state_list,batch_size=self.batch_size)
    new_q = rewards_list + self.gamma *  tf.reduce_max(new_qs_list,axis=1)
    new_q = new_q*(1-done_list)-done_list
    mask = tf.one_hot(action_list, num_actions)
    
    with tf.GradientTape() as tape:
        old_qs_list = self.model(old_state_list,training=False)
        q_action = tf.reduce_sum(tf.multiply(old_qs_list,mask),axis=1)
        loss = self.loss_function(new_q,q_action)
    grads = tape.gradient(loss,self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
# =============================================================================
#     old_state_list = np.array([transition[0] for transition in batch])#tf.experimental.numpy.vstack([tf.expand_dims(transition[0],axis=0) for transition in batch])
#     old_qs_list = self.model.predict(old_state_list,batch_size=self.batch_size)
#     new_state_list = np.array([transition[2]  if not None in transition[2] else  np.zeros((transition[0].shape)) for transition in batch])
#     new_qs_list = self.model_ref.predict(new_state_list,batch_size=self.batch_size)
#     
#     x = []
#     y = []
#     for index, (old_state, action, new_state, reward, done) in enumerate(batch):
#         if done:   # if this transition is from end_of_round
#             new_q = reward
#         else:
#             max_new_q = np.max(new_qs_list[index])
#             new_q = reward + self.gamma * max_new_q
# 
#         old_qs = old_qs_list[index]
#         old_qs[action] = new_q
# 
#         x.append(old_state)
#         y.append(old_qs)
# 
#     self.model.fit(np.array(x), np.array(y), batch_size=self.batch_size,verbose=0, shuffle=False)
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
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    action = ACTIONStonum.get(self_action)
    if None in old_state:
        pass
        
    else:
        if None in new_state:
            done = True
        else:
            done = False
        # Custom Events 
        
        if (action == 0 and self.lastaction == 2 and self.lastlastaction == 0) or (action == 2 and self.lastaction == 0 and self.lastlastaction == 2) or (action == 1 and self.lastaction == 3 and self.lastlastaction == 1) or (action == 3 and self.lastaction == 1 and self.lastlastaction == 3):
            events.append(LOOP)
        self.lastlastaction = self.lastaction
        self.lastaction = action
        
        
        if e.GOT_KILLED and not e.KILLED_SELF:
            events.append(KILLED_BY_OPPNENT)
        
        self.transitions.append(
            Transition(
                old_state,
                action,
                new_state,
                reward_from_events(self, events),
                done
                )
            )
        transit_count = len(self.transitions)
        self.counter +=1
        if self.counter % self.update_after_actions == 0 and transit_count > self.batch_size+2: 
            self.logger.debug(f"Starting updatemodel")
            train(self)
            self.logger.debug(f"Updated Model")
        if self.counter % self.update_target_network == 0:
            self.model_ref.set_weights(self.model.get_weights())
            self.logger.debug(f"Updated Target Model")
        
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
            reward_from_events(self, events),
            True)
        )

    # Store the model and analysis:

    tot_reward = 0
    for i in range(last_game_state['step']):
        tot_reward  += self.transitions[-i][3]
    self.total_rewards.append(tot_reward)
    self.episodelength.append(last_game_state['step'])
    
    with open("rewards.pt", "wb") as file:
        pickle.dump(self.total_rewards , file)
    with open("episodelength.pt", "wb") as file:
        pickle.dump(self.episodelength , file)

    tf.keras.models.save_model(self.model_ref,'model')
    
    self.checkpointcounter += 1
    if self.checkpointcounter % 200 == 0:
        avreward = np.mean(self.total_rewards[-50:-1])
        checkpoint = tf.train.Checkpoint(self.model)
        checkpoint.save(f'Checkpoints/checkpoint-eps_{self.checkpointcounter:03d}-rew_{avreward:.1f}')

        

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 10,
        e.INVALID_ACTION: -1,
        e.WAITED: -0.4,
        e.MOVED_DOWN: -0.1,
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.MOVED_UP: -0.1,
        e.BOMB_DROPPED: -0.3,
        e.CRATE_DESTROYED: 2,
        e.COIN_FOUND: 2,
        e.KILLED_SELF: -2,
        e.GOT_KILLED: -3,
        e.SURVIVED_ROUND: 1,
        LOOP: -1,
        KILLED_BY_OPPNENT: -5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum






