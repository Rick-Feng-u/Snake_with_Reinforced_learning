import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameRL, Direction, Point

MAX_MEMORY = 200_000 # can change
BATCH_SIZE = 1000 # can change
LR = 0.001 # can change

class Agent:

    def __init__(self):
        self.num_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0 # discount rate
        self.memeory = deque(maxlen = MAX_MEMORY) # popleft() if its exceed the memeory limit
        self.model = None # TODO :v
        self.trainer # TODO :v
        # TODO: model, trainer

    def getState(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int) # so bascially turning true = 1 and false = 0 (very big brain)

    def remember(self, state, action, reward, nextState, game_over):
        self.memeory.append((state, action, reward, nextState, game_over))

    def train_long_memeory(self):
        if len(self.memeory) > BATCH_SIZE: #make sure there is items to use
            mini_sample = random.sample(self.memeory, BATCH_SIZE) # liost of tuples
        else:
            mini_sample = self.memeory # just take the entire memoery if its smaller than the batch size

        stateS, actionS, rewardS, nextStateS, game_overS = zip(*mini_sample) # put everything states, actions ... together, could of used a for loop but this is easier in Pytorch
        #for state, action, reward, nextState, game_over in mini_sample:
        #    self.trainer.train_step(state, action, reward, nextState, game_over)

        self.trainer.train_step(stateS, actionS, rewardS, nextStateS, game_overS)

    def train_short_memeory(self, state, action, reward, nextState, game_over): # one step
        self.trainer.train_step(state, action, reward, nextState, game_over) # just train one game step

    def getAction(self,state):
        # trade off exploration / exploitation 
        # as the number of games getting larger, less random move is desired 

        self.epsilon = 80 - self.num_games # can change
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2) # this will give a random move to perform
            final_move[move] = 1
            #as the game train, epsilon gets smaller
            #and as the epsilon gets smaller its harder for random.randint(0, 200) to get a smaller number
            #threfore, there is less random moves
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model.predict(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

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

