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

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT",  "BOMB"]


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
    self.epsilon = 1#  start with random exploration
    self.epsilon_min = 0.1 
    self.epsilon_decay = 0.99 #  decay by .5% every episode => 1000 epsiodes = approx .6% random
    if self.train or not os.path.exists("model.pt"):
        self.logger.info("Setting up new model.")
    else:
        self.logger.info("Loading existing model.")
        with open("model.pt", "rb") as file:
            self.Q_table = pickle.load(file)

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
# =============================================================================
#     if self.train:
# =============================================================================
    if game_state['round'] != self.episode: # keep track with episodes
        # keep track of episode
        self.logger.info(f"New Episode{game_state['round']}")
        self.episode +=1
        # and update epsilon after each episode
        self.logger.info(f"self.epsilon{self.epsilon}")
        self.epsilon = self.epsilon*self.epsilon_decay
        
    if  (self.epsilon >  rand or self.epsilon_min  > rand) and self.train:
        self.logger.debug("Epsilon-greedy: Choosing action purely at random.")
        # Take random action
        action = int(np.random.choice(len(ACTIONS),1))
        self.logger.debug(f"Choosing action {action}.")
    else:
        self.logger.debug(f"state{state}")
        action = np.argmax(self.Q_table[state[0],state[1],state[2],state[3],state[4],:])
        self.logger.debug(f"Choosing action {action}.")
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
    if game_state is None:
        return [None]
     
    field =  game_state['field']
    x_self,y_self = game_state['self'][3]
    bombs = game_state['bombs']
    expmap = game_state['explosion_map']

    players = game_state['others']
    
    players_f = np.zeros(field.shape)
    for player in players:
        players_f.itemset(player[3],1)
    coins = np.zeros(field.shape)
    for coin in game_state['coins']:
        coins.itemset(coin,1)
    bombs_f = np.zeros(field.shape)
    for bomb in game_state['bombs']:
        bombs_f.itemset(bomb[0],1)
    features  = np.zeros(1+1+1+1+1) # up right down left center, 

    n=3
    
    expand_f = np.zeros(tuple(s+2*n for s in field.shape))    
    expand_f[tuple(slice(n,-n) for s in field.shape)] = field
    f_ac = expand_f[x_self+n-4:x_self+n+5,y_self+n-4:y_self+n+5]
    # feature 4
    # 0 = cant place bomb or selfkill, 1 can place bomb but destroy nothing, 2 bomb would destroys someting, 3 bomb would destroy alot
    selfkill = True
    expand_em = np.zeros(tuple(s+2*n for s in field.shape))    
    expand_em[tuple(slice(n,-n) for s in field.shape)] = expmap
    em_ac = expand_em[x_self+n-4:x_self+n+5,y_self+n-4:y_self+n+5]
    
    hypexpmap = np.zeros(em_ac.shape) #hypothetical explosion map if placed bomb
    for _ in range(4):
        obs = 0
        t= 0
        while t<4 and obs !=-1:
            if f_ac[4-t,4] == -1:
                obs ==-1
                break
            elif hypexpmap[4-t,4] != 0:
                pass
            else:
                hypexpmap[4-t,4] = -1
            t+=1
        hypexpmap = np.rot90(hypexpmap)
        f_ac = np.rot90(f_ac)
    selfkill = selfkill(f_ac,hypexpmap) #determine if bomb can be placed or selfkill inevitable
    # features 0 1 2 3:
    # up right down left: 0 == blocked, 1 == free, 2 == very good aka coin but not death 
    cycle_coords = [(x_self,y_self-1),(x_self+1,y_self),(x_self,y_self+1),(x_self-1,y_self)]
    for ind,coord in enumerate(cycle_coords):
        if field[coord] == 0 and bombs_f[coord] == 0 and players_f[coord] ==0 and expmap[coord] == 0:
            features[ind] = 1
            if (coins[coord] == 1): # if free and coin
                features[ind] = 2
    
    entexpmap = np.ones((17,17,4))
    for bomb in bombs: 
        entexpmap.itemset((bomb[0][0],bomb[0][1],bomb[1]),-1)
    
    if game_state['self'][2] and not selfkill:
        features[4] = 1
# =============================================================================
#         for dirr in direct:
#             features[direct] = 3
# =============================================================================
    return features.astype(int)


from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

def safespaces(f_ac,expmap):
    pathtodir = {
        (4,3):0,
        (5,4):1,
        (4,5):2,
        (3,4):3
        }
    f_ac_mod = np.abs(np.abs(f_ac)-1)
    ends = list(set(zip(*np.array(np.where(expmap == 0)).tolist())) & set(zip(*np.array(np.where((f_ac == 0))).tolist())))
    safespacedir = []
    pospath = []
    for endn in ends:
        grid = Grid(matrix= f_ac_mod)
        start = grid.node(4, 4)
        finder = AStarFinder()
        end = grid.node(endn[1],endn[0])
        path, runs = finder.find_path(start, end, grid)
        if len(path)> 1 and len(path)<6:
            pospath.append(path)
    if pospath != []:
        for ppath in pospath:
            safespacedir.append(pathtodir.get(ppath[1]))
        return False#,safespacedir
    return True#,None

def selfkill(f_ac,hypexpmap):
    f_ac_mod = np.abs(np.abs(f_ac)-1)
    ends = list(set(zip(*np.array(np.where(hypexpmap == 0)).tolist())) & set(zip(*np.array(np.where((f_ac == 0))).tolist())))
    for endn in ends:
        grid = Grid(matrix= f_ac_mod)
        start = grid.node(4, 4)
        finder = AStarFinder()
        end = grid.node(endn[1],endn[0])
        path, runs = finder.find_path(start, end, grid)
        if len(path)> 1 and len(path)<6:
            return False
    return True



def state_to_features2(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in vironment.py to see
    it contains.
     
    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return [None]
    coins=game_state['coins']
    bomb = game_state['bombs']
    explosion_map = game_state['explosion_map']
    us = game_state['self']
    field = game_state['field']
    ownpos = list(game_state['self'][3])
    crates= np.array(np.where((field ==1)))
    if coins != [] and crates.size >0:      
        nearest_coin_rel = np.array(coins[np.argmin(np.linalg.norm(np.array(coins)-np.array(ownpos),axis=1))])-np.array(ownpos)
    else:
        nearest_coin_rel =[0,0]
    if crates.size>0:
        nearest_crate_rel =  np.array(crates[:, np.argmin(np.linalg.norm(crates-np.array(ownpos)[:,None],axis=0))])-np.array(ownpos)
    else:
        nearest_crate_rel =[0,0]
    
    # for i in range(2): 
    #pos= field[ownpos[0]-1:ownpos[0]+2,ownpos[1]-1:ownpos[1]+2]
    pos = np.array([field[ownpos[0]+1, ownpos[1]],field[ownpos[0]-1, ownpos[1]], field[ownpos[0], ownpos[1]+1], field[ownpos[0], ownpos[1]-1]] )
    
    
    
    #save = np.array([])
    #for i in range(4):
        
        
    #mod_pos = [game_state["self"][3][0] % 2, game_state["self"][3][1] % 2]
    return np.concatenate((nearest_coin_rel, nearest_crate_rel, pos)) # np.array(pos).flatten()))
