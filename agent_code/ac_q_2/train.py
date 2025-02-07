# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 11:13:26 2022

@author: ztec1
"""
import pickle

from collections import namedtuple, deque
from typing import List
import os
import events as e

from .callbacks import state_to_features
from .callbacks import ACTIONS
import numpy as np


# This is only an example!
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 400 # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
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
    #self.logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    
    if os.path.exists('model.pt'):
        self.logger.info("Retraining from saved state.")
        self.logger.info("Loading model.")
        with open("model.pt", "rb") as file:
            self.Q_table = pickle.load(file)
    else:
        self.Q_table =  np.zeros((4,4,4,4,2,  num_actions ))
    self.alpha = 0.1
    self.gamma = 0.9
# Analysis stuff
    self.total_rewards = []
    self.episodelength = []
    
# python main.py play --agents ac_q_1 --train 1 --no-gui  --scenario classic --n-rounds 100
# python main.py play --agents ac_q_1 --train 0  --scenario classic --n-rounds 1 


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
    old_state,action,new_state,reward = self.transitions[-1]
    if None in old_state:
        pass
    else:
        q_val = self.Q_table[old_state[0],old_state[1],old_state[2],old_state[3],old_state[4],  action]
        if None in new_state: #if finished
            self.Q_table[old_state[0],old_state[1],old_state[2],old_state[3],old_state[4], action] = 0
        else: # update q-table
        # calculate q value for  given state and  action
            max_val = np.max(self.Q_table[new_state[0],new_state[1],new_state[2],new_state[3],new_state[4],:])
            new_q_val = (1 - self.alpha) * q_val + self.alpha * (reward + self.gamma * max_val)
            self.Q_table[old_state[0],old_state[1],old_state[2],old_state[3],old_state[4], action] = new_q_val


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
# =============================================================================
#     self.logger.debug(f'{self.transitions}')
#     for old_state,action,new_state,reward in self.transitions:
#         self.logger.debug(f'{old_state,action,new_state,reward}')
#         if None in old_state:
#             pass
#         else:
#             q_val = self.Q_table[old_state[0],old_state[1],old_state[2],old_state[3],old_state[4],  action]
#             if None in new_state: #if finished
#                 self.Q_table[old_state[0],old_state[1],old_state[2],old_state[3],old_state[4], action] = 0
#             else: # update q-table
#             # calculate q value for  given state and  action
#                 max_val = np.max(self.Q_table[new_state[0],new_state[1],new_state[2],new_state[3],new_state[4],:])
#                 new_q_val = (1 - self.alpha) * q_val + self.alpha * (reward + self.gamma * max_val)
#                 self.Q_table[old_state[0],old_state[1],old_state[2],old_state[3],old_state[4], action] = new_q_val
# 
# =============================================================================
    # Store the model
    with open(r"model.pt", "wb") as file:
        pickle.dump(self.Q_table, file)
        
    # Calculate total reward for analysis
    tot_reward = 0
    for transit in self.transitions:
        tot_reward  += transit[3]
    self.total_rewards.append(tot_reward)
    self.episodelength.append(last_game_state['step'])
    with open("rewards.pt", "wb") as file:
        pickle.dump(self.total_rewards , file)
    with open("episodelength.pt", "wb") as file:
        pickle.dump(self.episodelength , file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 2,
        e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -1,
        e.WAITED: -0.3,
        e.KILLED_SELF: -5,
        e.BOMB_DROPPED: -0.2,
        e.COIN_FOUND: 0.4,
        e.CRATE_DESTROYED: 1,
        
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum






