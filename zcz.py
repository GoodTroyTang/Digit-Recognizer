# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

class optim_conv(nn.Module):
    def __init__(self, weight):
        super(optim_conv, self).__init__()
        self.weight = Variable(torch.from_numpy(weight), requires_grad=False)
        self.conv1 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        
        
    def forward(self, input):
        h = nn.functional.conv2d(input, self.weight, padding=1)
        output = self.conv1(h)
        return output
        

if __name__ == '__main__':
    iternum = 100
    weight = np.random.randn(3, 3)
    model = optim_conv(weight)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    
    for iteridx in range(iternum):
        optimizer.zero_grad()                           # clear gradients for this training step
        image = np.random.randn(50, 5)
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)
        input = Variable(torch.from_numpy(image), requires_grad=False)
        target = Variable(torch.from_numpy(image), requires_grad=False)
        output = model(input)
        loss = criterion(target, output)
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step() 