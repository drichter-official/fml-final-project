import sys
from io import StringIO
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List

import gym
import pygame as pygame
from gym import spaces
from gym.core import ObsType, ActType
import numpy as np

# Game properties
# from gym_bomberman.environments import Renderer
from gym_bomberman.environments.agent import Agent
from enum import IntEnum

# Arena
COLS = 17
ROWS = 17

# GUI
GRID_SIZE = 30
WIDTH = 1000
HEIGHT = 600
GRID_OFFSET = [(HEIGHT - ROWS * GRID_SIZE) // 2] * 2
ASSET_DIR = Path(__file__).parent.parent.parent / "assets"

FPS = 2


class Actions(IntEnum):
    # Actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    BOMB = 4
    WAIT = 5


class TileType(IntEnum):
    EXPLOSION = -3
    BOMB = -2
    STONE_WALL = -1
    FREE = 0
    COIN = 1
    CRATE = 2
    AGENT = 3


class CoinEnv(gym.Env[int, int]):
    """
    ### Description
    The CoinEnv includes is a field with some coins on it. By walking to a coin, the agent can collect it.
    
    ### Action space
    An agent can either go up, down, left or right, can set a bomb at the current field or wait.
    
    ### Observation space
    The observation space gives information about the entire field.
    Each field can have on of the following values:
        - -1: stone wall
        - 0: free
        - 1: coin
    """

    def __init__(self, rows=ROWS, cols=COLS):
        self.agent = None
        self.screen = None

        self.frame = 0
        self.arena_rows = rows
        self.arena_cols = cols

        # Agent can wait and go left/right/up/down
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.arena_rows, self.arena_cols),
                                            dtype=np.int8)

        self.done = True
        self.reset()

    def _generate_arena(self):
        # Each field contains a coin with 5% probability
        self.arena = np.random.choice([TileType.FREE, TileType.COIN], size=(self.arena_rows, self.arena_cols),
                                      p=[0.95, 0.05])

        # Set walls
        self.arena[:1, :] = TileType.STONE_WALL
        self.arena[-1:, :] = TileType.STONE_WALL
        self.arena[:, :1] = TileType.STONE_WALL
        self.arena[:, -1:] = TileType.STONE_WALL

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        assert (self.action_space.contains(action))

        reward = 0
        x, y = (self.agent.x, self.agent.y)

        movement_vector = (0, 0)
        if action == Actions.UP:
            movement_vector = (0, -1)
        elif action == Actions.DOWN:
            movement_vector = (0, 1)
        elif action == Actions.LEFT:
            movement_vector = (-1, 0)
        elif action == action == Actions.RIGHT:
            movement_vector = (1, 0)
        elif action == Actions.WAIT:
            pass
        elif action == Actions.BOMB:
            # TODO
            pass
        else:
            # Penalize invalid move
            reward -= 5

        mov_x, mov_y = movement_vector
        if self.arena[x + mov_x, y + mov_y] == TileType.FREE:
            self.arena[x, y] = TileType.FREE
            self.agent.x += mov_x
            self.agent.y += mov_y
            next_agent_location = (x + mov_x, y + mov_y)
        else:
            # Penalize invalid move against wall
            reward -= 1
            next_agent_location = (x, y)

        if self.arena[next_agent_location] == TileType.COIN:
            reward += 20
        self.arena[next_agent_location] = TileType.AGENT

        self.frame += 1

        coins = self.coins
        print(coins)
        if self.frame > 500 or len(self.coins) == 0:
            self.done = True

        return self.arena, reward, self.done, {}

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> Union[
        ObsType, tuple[ObsType, dict]]:
        self.frame = 0
        self._generate_arena()
        possible_starting_positions = [
            (1, 1),
            (1, self.arena_rows - 2),
            (self.arena_cols - 2, 1),
            (self.arena_cols - 2, self.arena_rows - 2)]
        x, y = random.choice(possible_starting_positions)
        self.agent = Agent(0, x, y)
        self.arena[x, y] = TileType.AGENT
        self.done = False

        return self.arena

    def render(self, mode="human"):
        if mode == "human":
            # if self.screen is None:
            #     self._init_rendering()
            # # Reset to black background
            # self.screen.blit(self.background, (0, 0))
            #
            # # Render map
            # for x in range(self.arena.shape[1]):
            #     for y in range(self.arena.shape[0]):
            #         pos = (self.grid_offset[0] + GRID_SIZE * x, self.grid_offset[1] + GRID_SIZE * y)
            #
            #         if self.arena[x, y] == STONE_WALL:
            #             self.screen.blit(self.t_wall, pos)
            #
            #         if self.arena[x, y] == COIN:
            #             self.screen.blit(self.t_coin, pos)
            #
            #         if self.arena[x, y] == AGENT:
            #             self.screen.blit(self.t_agent_blue, pos)
            #
            # # Render frame count
            # self.render_text(f'Frame {self.frame:d}', self.grid_offset[0], HEIGHT - self.grid_offset[1] / 2,
            #                  (64, 64, 64),
            #                  v_align='center', h_align='left', size='medium')
            #
            # pygame.display.update()
            # pygame.time.wait(1000 // FPS)

            outfile = sys.stdout
            return self._to_text(outfile)
        elif mode == "rgb_array":
            pass
        elif mode == "ansi":
            outfile = StringIO()
            return self._to_text(outfile)

    def _to_text(self, outfile):
        for x in range(self.arena.shape[1]):
            for y in range(self.arena.shape[0]):
                element = self.arena[y, x]
                outfile.write("{}".format(["💥", "💣", "❌", "👣", "❎", "🎁", "😎"][element + 3]))
            outfile.write('\n')
        return outfile

    def _init_rendering(self):
        self.clock = pygame.time.Clock()
        pygame.init()

        self.grid_offset = [(HEIGHT - self.arena_rows * GRID_SIZE) // 2] * 2
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('BombeRLe')
        icon = pygame.image.load(ASSET_DIR / f'bomb_yellow.png')
        pygame.display.set_icon(icon)

        # Background and tiles
        self.background = pygame.Surface((WIDTH, HEIGHT))
        self.background = self.background.convert()
        self.background.fill((0, 0, 0))
        self.t_wall = pygame.image.load(ASSET_DIR / 'brick.png')
        self.t_crate = pygame.image.load(ASSET_DIR / 'crate.png')
        self.t_coin = pygame.image.load(ASSET_DIR / 'coin.png')
        self.t_agent_blue = pygame.image.load(ASSET_DIR / 'robot_blue.png')

        # Font for scores and such
        font_name = ASSET_DIR / 'emulogic.ttf'
        self.fonts = {
            'huge': pygame.font.Font(font_name, 20),
            'big': pygame.font.Font(font_name, 16),
            'medium': pygame.font.Font(font_name, 10),
            'small': pygame.font.Font(font_name, 8),
        }

    def render_text(self, text, x, y, color, h_align='left', v_align='top', size='medium', aa=False):
        text_surface = self.fonts[size].render(text, aa, color)
        text_rect = text_surface.get_rect()
        if h_align == 'left':
            text_rect.left = x
        if h_align == 'center':
            text_rect.centerx = x
        if h_align == 'right':
            text_rect.right = x
        if v_align == 'top':
            text_rect.top = y
        if v_align == 'center':
            text_rect.centery = y
        if v_align == 'bottom':
            text_rect.bottom = y

        self.screen.blit(text_surface, text_rect)

    @property
    def coins(self) -> List[Tuple[int, int]]:
        return list(zip(*np.where(self.arena == TileType.COIN)))

    @property
    def crates(self) -> List[Tuple[int, int]]:
        return list(zip(*np.where(self.arena == TileType.CRATE)))

    @property
    def bombs(self) -> List[Tuple[int, int]]:
        return list(zip(*np.where(self.arena == TileType.BOMB)))

    @property
    def explosions(self) -> List[Tuple[int, int]]:
        return list(zip(*np.where(self.arena == TileType.EXPLOSION)))

    def standing_on_coin(self, x, y) -> bool:
        coins = self.coins
        return (x, y) in coins


if __name__ == "__main__":
    benv = CoinEnv()
    benv.render()
    benv.step(Actions.RIGHT)
    benv.render()
    benv.step(Actions.UP)
    benv.render()
    benv.step(Actions.LEFT)
    benv.render()
    benv.step(Actions.WAIT)
    benv.render()
    benv.step(Actions.LEFT)
    benv.render()
    benv.step(Actions.DOWN)
    benv.render()
    benv.step(Actions.DOWN)
    benv.render()
