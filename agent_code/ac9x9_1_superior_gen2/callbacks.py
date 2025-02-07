# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:39:46 2022

@author: danie
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 09:41:16 2022

@author: ztec1
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

import agent_code.ac9x9_1_superior_gen2.Setting as S
from agent_code.rule_based_agent.callbacks import act as rule_based_act
from agent_code.rule_based_agent.callbacks import setup as rule_based_setup

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


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
        
    self.epsilon = S.EPSILON_START# start with 100% random exploration
    self.epsilon_min = S.EPSILON_MIN # while training explore at  least  10% of steps
    self.epsilon_decay = S.EPSILON_DECAY
    
    self.epsilon_imitate = S.EPSILON_IMITATE # percentage of random moves get imitatet
    self.epsilon_imitate_decay = S.EPSILON_IMITATE_DECAY
    
    self.softmax = S.SOFTMAX
    self.imitate = S.IMITATE
        
    if self.imitate:
        imitate_setup(self)
    
    if True:
        self.logger.info("Loading Checkpoint")
       # self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),loss=tf.keras.losses.Huber())
        self.model = create_q_model()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=S.LEARNINGRATE, clipnorm=1.0),loss=tf.keras.losses.Huber())
        print("loading checkpoints")
        self.model.load_weights("C:/Users/danie/Meine Ablage/Github/bomberman_rl/agent_code/ac9x9_2_superior/BestCheck/checkpoint-eps_4800-rew_51.9-1")
        #self.model.load_weights("C:/Users/danie/Meine Ablage/Github/bomberman_rl/agent_code/ac9x9_1_superior/BestCheck/checkpoint-eps_5000-rew_59.0-1")
        #self.model.load_weights("C:/Users/danie/Meine Ablage/Github/bomberman_rl/agent_code/ac9x9_2_superior/BestCheck/checkpoint-eps_12200-rew_53.7-1")
        print("Success")
    else: 
        if self.train or not os.path.isdir("model"):
            self.logger.info("Setting up new model.")
        else:
            self.logger.info("Loading model.")
            self.model = tf.keras.models.load_model('model')
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=S.LEARNINGRATE, clipnorm=1.0),loss=tf.keras.losses.Huber())
            
    self.model_cc = tf.keras.models.load_model('model_cc')
    self.model_cc.summary()

def create_q_model():
    input_shape = (9,9,4)
    model = keras.Sequential()
    model.add(layers.Conv2D(16, kernel_size=(5, 5), input_shape=input_shape))
    model.add(layers.Conv2D(32 , kernel_size=(3, 3), padding='valid',activation='relu', ))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='valid',activation='relu', ))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(6, activation='linear'))
    model.summary()
    return model
# python main.py play --agents task2_alt --train 0 --scenario coin-heaven --no-gui --n-rounds 10


def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    if game_state['round'] != self.episode: # keep track with episodes
        # keep track of episode
        self.logger.info(f"Starting Episode {game_state['round']}")
        self.episode +=1
        # and update epsilon after each episode
        self.epsilon = self.epsilon*self.epsilon_decay
        self.epsilon_imitate = self.epsilon_imitate*self.epsilon_imitate_decay
    
    self.logger.info('Predicting Action')
    features0,features1,features2,features3 = state_to_features(game_state)
    cc_features,coin = state_to_features_cc(game_state)
    feature_tensor0 = tf.expand_dims(features0, 0)
    feature_tensor1 = tf.expand_dims(features1, 0)
    feature_tensor2 = tf.expand_dims(features2, 0)
    feature_tensor3 = tf.expand_dims(features3, 0)
    cc_tensor = tf.expand_dims(cc_features, 0)
    rand = np.random.rand(1)[0]
    action = 0
    
    if (self.epsilon >  rand or self.epsilon_min  > rand) and self.train:
        if self.epsilon_imitate > rand and self.imitate:
            if not None in features0:
                Action = rule_based_act(self, game_state)
                if Action is not None:
                    self.logger.debug(f"Choosing action {Action}.")
                    return Action  
        if not self.softmax:
            self.logger.debug("Epsilon-greedy: Choosing action purely at random")
            action = int(np.random.choice(len(ACTIONS),1))
        else:   
            action_probs = self.model(feature_tensor0,training=False)
            action_probs,_ = tf.linalg.normalize(action_probs)
            action_probs = tf.keras.activations.softmax(5*action_probs)
            self.logger.debug(f"Using Softmax policy: Choosing action with probability{action_probs}")
            action = np.dot(np.arange(len(ACTIONS)),np.random.multinomial(1,action_probs[0]))
        self.logger.debug(f"Choosing action {ACTIONS[action]}.")
    else:
        action_probs0 = self.model(feature_tensor0,training=False)[0]
        action_probs0 = action_probs0/sum(action_probs0)
        
        action_probs1 = perm(self.model(feature_tensor1,training=False)[0])
        action_probs1 = action_probs1/sum(action_probs1)

        action_probs2 = perm(perm(self.model(feature_tensor2,training=False)[0]))
        action_probs2 = action_probs2/sum(action_probs2)

        action_probs3 = perm(perm(perm(self.model(feature_tensor3,training=False)[0])))
        action_probs3 = action_probs3/sum(action_probs3)

        action_probs = np.sum(np.vstack((action_probs0,action_probs1,action_probs2,action_probs3)),axis=0)
        
        
        self.logger.debug(f"Using argmax policy {action_probs}")
        
        action = np.argmax(action_probs)    
        
        self.logger.debug(f"Choosing action {ACTIONS[action]}.")
    return ACTIONS[action]

k1 = {
      0:1,
      1:2,
      2:3,
      3:0      
      }
def perm(actions): # 0 1 2 3 to 3 0 1 2 
    act45 = actions[-2:]
    newact = np.zeros(4)
    for k in range(4): 
        newact[k] = actions[k1.get(k)]
    return np.concatenate((newact,act45))

def imitate_setup(self):
    rule_based_setup(self)
def imitate_act(self, game_state):
    return rule_based_act(self, game_state)

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
    field = game_state['field']
    exp_map = game_state['explosion_map']
    ownpos = list(game_state['self'][3])
    
    n=3
    
    expand_f = np.zeros(tuple(s+6 for s in field.shape))    
    expand_f[tuple(slice(n,-n) for s in field.shape)] = field
    f_ac = expand_f[ownpos[0]+n-4:ownpos[0]+n+5,ownpos[1]+n-4:ownpos[1]+n+5]

    expand_em = np.zeros(tuple(s+6 for s in field.shape))    
    expand_em[tuple(slice(n,-n) for s in field.shape)] = exp_map
    em_ac = expand_em[ownpos[0]+n-4:ownpos[0]+n+5,ownpos[1]+n-4:ownpos[1]+n+5]
    
    expand_cab = np.zeros(tuple(s+6 for s in field.shape)) 
    coinsandbombs = np.zeros(field.shape)
    for coin in game_state['coins']:
        coinsandbombs.itemset(coin,1)
    for bomb in game_state['bombs']:
        coinsandbombs.itemset(bomb[0],-1)#-1/4*bomb[1])

    expand_cab[tuple(slice(n,-n) for s in field.shape)] = coinsandbombs
    cab_ac = expand_cab[ownpos[0]+n-4:ownpos[0]+n+5,ownpos[1]+n-4:ownpos[1]+n+5]
    
    expand_sap = np.zeros(tuple(s+6 for s in field.shape)) 
    selfandplayers = np.zeros(field.shape)
    if game_state['self'][2]:
        selfandplayers.itemset(game_state['self'][3],1)
        
    for player in game_state['others']:
        selfandplayers.itemset(player[3],-1)     
        
    expand_sap[tuple(slice(n,-n) for s in field.shape)] = selfandplayers
    sap_ac = expand_sap[ownpos[0]+n-4:ownpos[0]+n+5,ownpos[1]+n-4:ownpos[1]+n+5]
    
    # print(f_ac,em_ac,cab_ac,sap_ac)
    
    return np.stack((f_ac,em_ac,cab_ac,sap_ac),axis=-1),np.stack((np.rot90(f_ac),np.rot90(em_ac),np.rot90(cab_ac),np.rot90(sap_ac)),axis=-1),np.stack((np.rot90(f_ac,k=2),np.rot90(em_ac,k=2),np.rot90(cab_ac,k=2),np.rot90(sap_ac,k=3)),axis=-1),np.stack((np.rot90(f_ac,k=3),np.rot90(em_ac,k=3),np.rot90(cab_ac,k=3),np.rot90(sap_ac,k=3)),axis=-1)

def state_to_features_cc(game_state: dict) -> np.array:
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
    ownpos = list(game_state['self'][3])
    if coins != []:      
        nearest_coin_rel = np.array(coins[np.argmin(np.linalg.norm(np.array(coins)-np.array(ownpos),axis=1))])-np.array(ownpos)
        if np.linalg.norm(nearest_coin_rel) != 0:
            nearest_coin_norm = nearest_coin_rel/np.linalg.norm(nearest_coin_rel)
        else:
            nearest_coin_norm =[0,0]
    else:
        nearest_coin_norm =[0,0]

    mod_pos = [ownpos[0] % 2, ownpos[1] % 2]
    if coins!= []:
        return np.concatenate((nearest_coin_norm,mod_pos)),True
    else:
        return np.concatenate((nearest_coin_norm,mod_pos)),False


