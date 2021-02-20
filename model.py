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