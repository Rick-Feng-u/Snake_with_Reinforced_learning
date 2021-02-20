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
        self.num_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0 # discount rate
        self.memeory = deque(maxlen = MAX_MEMORY) # popleft() if its exceed the memeory limit
        # TODO: model, trainer

    def getState(self, game):
        pass

    def remember(self, state, action, reward, nextState, game_over):
        pass

    def train_long_memeory(self):
        pass

    def train_short_memeory(self, state, action, reward, nextState, game_over): # one step
        pass

    def getAction(self,state):
        pass

def train():
    plot_score = [] # for plotting
    plot_mean_score = []
    total_score = 0
    record = 0 # best score
    agent = Agent()
    game = SnakeGameRL()
    while True:
        # get old state 
        stateOld = agent.getState(game)

        #get move based on this current state
        final_move =  agent.getAction(stateOld)

        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        stateNew = agent.getState(game)

        #train short memoery 
        agent.train_short_memeory(stateOld,final_move,reward,stateNew,game_over)

        #remember
        agent.remember(stateOld,final_move,reward,stateNew,game_over)

        if game_over:
            #train the long memoery, and plot the result
            game.reset()
            agent.num_games += 1
            agent.train_long_memeory()

            if score > record:
                record = score
                # TODO agent.model.save()

            print("Game: ", agent.num_games, "Score: ", score, "highest score currently: ", record)

            # TODO plotting


if __name__ == '__main__':
    train()

