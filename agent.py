import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameRL, Direction, Point

MAX_MEMORY = 200_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        pass

    def getState(self, game):
        pass

    def remember(self, state, action, reward, nextState, game_over):
        pass

    def train_long_memeory(self):
        pass

    def train_short_memeory(self): # one step
        pass

    def getAction(self,state):
        pass