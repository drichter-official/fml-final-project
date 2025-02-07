from collections import defaultdict
from typing import List


class Agent:
    id: int
    points: int
    dead: bool
    bombs_left: bool
    statistics: defaultdict[int]
    events: list

    x: int
    y: int

    def __init__(self, identifier: int, x: int, y: int):
        self.x = x
        self.y = y
        self.id = identifier
        self.points = 0

        self.dead = False
        self.score = 0

        self.statistics = defaultdict(int)
        self.trophies = []

        self.events = []
        self.bombs_left = True

        self.last_game_state = None
        self.last_action = None

    def update_score(self, points: int) -> None:
        self.score += points
