import sys
from pathlib import Path

import pygame

# GUI properties

GRID_SIZE = 30
WIDTH = 1000
HEIGHT = 600
ASSET_DIR = Path(__file__).parent.parent.parent / "assets"

EXPLOSION = -3
BOMB = -2
STONE_WALL = -1
FREE = 0
COIN = 1
CRATE = 2
AGENT = 3

FPS = 2


class Renderer:
    def __init__(self, env):
        self.env = env
        self.clock = pygame.time.Clock()
        pygame.init()

        self.grid_offset = [(HEIGHT - env.arena_rows * GRID_SIZE) // 2] * 2
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

    def render(self):
        # Reset to black background
        self.screen.blit(self.background, (0, 0))

        # Render map
        outfile = sys.stdout
        for x in range(self.env.arena.shape[1]):
            for y in range(self.env.arena.shape[0]):
                pos = (self.grid_offset[0] + GRID_SIZE * x, self.grid_offset[1] + GRID_SIZE * y)

                if self.env.arena[x, y] == STONE_WALL:
                    self.screen.blit(self.t_wall, pos)

                if self.env.arena[x, y] == COIN:
                    self.screen.blit(self.t_coin, pos)

                if self.env.arena[x, y] == AGENT:
                    self.screen.blit(self.t_agent_blue, pos)

        # Render frame count
        self.render_text(f'Frame {self.env.frame:d}', self.grid_offset[0], HEIGHT - self.grid_offset[1] / 2,
                         (64, 64, 64),
                         v_align='center', h_align='left', size='medium')

        pygame.display.update()
        pygame.time.wait(1000 // FPS)

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
