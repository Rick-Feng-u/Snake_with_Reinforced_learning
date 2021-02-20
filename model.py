import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os 

class Linear_QNet(nn.Module):
    def __init__(self, input_size, output_size, ):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, Xtensor):
        Xtensor = F.relu(self.linear1(Xtensor))
        Xtensor = self.linear2(Xtensor)
        return Xtensor

    def save(self, file_name = 'model.pth'):
        modelFolderPath = './model'
        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)
        
        file_name = os.path.join(modelFolderPath, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, LR, gamma):
        self.LR = LR
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.LR)
        self.crieria = nn.MSELoss()

    def train_setp(self, state, action, reward, New_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        #(n, x) n is the number of batch, for the long memoery is already passed in as a tuple, so we dont need to change it

        if len(state.shape) == 1: #however, if this is a short memoery, we need to make them into tuples
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # this is how you define tuple with one value

        #now we impelment Bellman equation for model
        # 1. predicted Q value using current state
        predi = self.model(state)

        target = predi.clone()
        for index in range(len(game_over)):
            Q_new = reward[index]
            if not game_over[index]:
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))

            target[index][torch.argmax(action).item()] = Q_new

        # 2. Q_new = reward + gamma * max(next predicted Q value) --- only do this if not over
        # predi.clone()
        # predis[argmax(action)] = Q_new

        self.optimizer.zero_grad()
        loss = self.crieria(target, predi)
        loss.backward()

        self.optimizer.step()


