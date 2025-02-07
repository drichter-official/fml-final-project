from typing import Dict, List

import gym
import numpy as np
from gym import spaces

from items import Coin, Bomb, Explosion

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_BOMB = 4
ACTION_WAIT = 5

EXPLOSION = -3
BOMB = -2
STONE_WALL = -1
FREE = 0
COIN = 1
CRATE = 2
AGENT = 3


class BombermanGymEnv(gym.Env):
    running: bool = False
    step: int
    replay: Dict
    round_statistics: Dict

    # agents: List[Agent]
    # active_agents: List[Agent]
    arena: np.ndarray
    coins: List[Coin]
    bombs: List[Bomb]
    explosions: List[Explosion]

    round_id: str

    def __init__(self, width: int, height: int):
        self.frame = 0
        self.round_id = 0
        self.arena = np.zeros((height, width), dtype=np.int8)

        self.frame = 0
        self.arena_rows = width
        self.arena_cols = height

        # Agent can wait and go left/right/up/down
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-EXPLOSION, high=AGENT,
                                            shape=(self.arena_rows, self.arena_cols),
                                            dtype=np.int8)

    def coins(self):
        return zip(*np.where(self.arena == COIN))

    def crates(self):
        return zip(*np.where(self.arena == CRATE))

    def bombs(self):
        return zip(*np.where(self.arena == BOMB))

    def explosions(self):
        return zip(*np.where(self.arena == EXPLOSION))



