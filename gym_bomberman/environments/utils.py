from typing import Tuple
import numpy as np
from agents import Agent

from items import Coin

def build_arena(
    cols: int, 
    rows: int, 
    crate_density: float,
    num_coins: int,
    agents: list[Agent],
    rng: np.random.Generator) -> Tuple[np.ndarray, list[Coin], list[Agent]]:

    WALL = -1
    FREE = 0
    CRATE = 1

    arena = np.zeros((cols, rows), int)
    # Crates in random locations
    arena[rng.random((cols, rows)) < crate_density] = CRATE
    # Walls
    arena[:1, :] = WALL
    arena[-1:, :] = WALL
    arena[:, :1] = WALL
    arena[:, -1:] = WALL
    for x in range(cols):
        for y in range(rows):
            if (x + 1) * (y + 1) % 2 == 1:
                arena[x, y] = WALL
    # Clean the start positions
    start_positions = [(1, 1), (1, rows - 2), (cols - 2, 1), (cols - 2, rows - 2)]
    for (x, y) in start_positions:
        for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if arena[xx, yy] == 1:
                arena[xx, yy] = FREE
    # Place coins at random, at preference under crates
    coins = []
    all_positions = np.stack(np.meshgrid(np.arange(cols), np.arange(rows), indexing="ij"), -1)
    crate_positions = rng.permutation(all_positions[arena == CRATE])
    free_positions = rng.permutation(all_positions[arena == FREE])
    coin_positions = np.concatenate([
        crate_positions,
        free_positions
    ], 0)[:num_coins]
    for x, y in coin_positions:
        coins.append(Coin((x, y), collectable=arena[x, y] == FREE))
    # Reset agents and distribute starting positions
    active_agents = []
    for agent, start_position in zip(agents, rng.permutation(start_positions)):
        active_agents.append(agent)
        agent.x, agent.y = start_position
    return arena, coins, active_agents