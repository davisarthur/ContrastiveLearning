import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Simple convolutional neural network for embedding the fashion MNIST dataset
class Network(nn.Module):
  def __init__(self):
    super().__init__()

    # convolutional layers 
    self.conv_layers = nn.Sequential(
        nn.Conv2d(1, 20, kernel_size=7, padding=3),
        #nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(20, 40, kernel_size=3, padding=1),
        #nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(40, 80, kernel_size=3, padding=1),
        #nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(80, 160, kernel_size=3, padding=1),
        #nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.fc1 = nn.Linear(in_features=160, out_features=80)
    self.fc2 = nn.Linear(in_features=80, out_features=80)

  def forward(self, t):
    # conv 1
    t = self.conv_layers(t)
    # fc1
    t = t.view(t.size(0), -1)
    t = torch.nn.functional.normalize(t)
    t = self.fc1(t)
    t = F.relu(t)

    # fc2
    t = self.fc2(t)
    t = torch.nn.functional.normalize(t)
    return t
  
  def embed(self, t):
    t = self.conv_layers(t)

    # fc1
    t = t.view(t.size(0), -1)
    t = torch.nn.functional.normalize(t)
    return t