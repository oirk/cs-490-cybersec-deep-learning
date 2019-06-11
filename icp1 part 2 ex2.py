from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.fc1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(64, 10)
        
    def forward(self, x):
        # Hidden layer with reLU activation
        x = F.relu(F.relu(self.hidden2(self.fc1(x))))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)
        
        return x


model = Network()
model
