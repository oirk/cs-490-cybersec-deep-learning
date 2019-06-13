# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 23:12:20 2019

@author: colby chandler
"""

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

import helper

import os
from torch import nn, optim
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(150528, 752)
        self.fc2 = nn.Linear(752, 376)
        self.fc3 = nn.Linear(376, 188)
        self.fc4 = nn.Linear(188, 94)
        self.fc5 = nn.Linear(94,47)
        self.fc6 = nn.Linear(47,23)
        self.fc7 = nn.Linear(23,11)
        self.fc8 = nn.Linear(11,2)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))

        # output so no dropout here
        x = F.log_softmax(self.fc8(x), dim=1)

        return x


data_dir = 'Cat_Dog_data/Cat_Dog_data'

transform = transform = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])
    
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])

dataset = datasets.ImageFolder(os.path.join(data_dir), transform=transform)

dataloader = dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)



train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)


images, labels = next(iter(trainloader))

model = Classifier()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 5
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
  
        running_loss += loss.item()
       
        
    else:
        test_loss = 0
        accuracy = 0
        ## TODO: Implement the validation pass and print out the validation accuracy
        with torch.no_grad():
            model.eval()
    # validation pass here
            for images, labels in testloader:
                log_ps = model(images)
                test_loss += criterion(log_ps,labels)

                ps = torch.exp(log_ps)
                top_p,top_class = ps.topk(1,dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))
            print ( "Epoch: {}/{}.. ".format(e+1,epochs),
                    "Training Loss: {:.3F}.. ".format(running_loss/len(testloader)),
                    "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                    "Test Accuracy: {:.3f}.. ".format(accuracy/len(testloader)))
            model.train()

    
print ("done")