#!/usr/bin/env python
# coding: utf-8

# In[171]:


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time


# In[209]:


repo = 'C:\\Users\Pulkit Dixit\Desktop\IE 534\HW3'
batch_size = 100
learn_rate = 0.001
scheduler_step_size = 10
scheduler_gamma = 0.1
num_epochs = 15


# In[210]:


transform_train = transforms.Compose([#transforms.Resize((16,16)),
                                      transforms.RandomRotation(10),
                                      transforms.RandomHorizontalFlip(),
                                      #transforms.RandomVerticalFlip(),
                                      transforms.ToTensor()
                                     ])


# In[211]:


train_dataset = torchvision.datasets.CIFAR10(root = '~/scratch/', train=True, transform=transform_train, download=False)
test_dataset = torchvision.datasets.CIFAR10(root = '~/scratch/', train=False, transform=transform_train, download=False)


# In[212]:


#print(train_dataset[1][1])
#print(train_dataset[0][0].size())


# In[213]:


train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True, num_workers = 8)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=False, num_workers = 8)


# In[159]:


#print(train_loader)


# In[214]:


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 4, stride = 1, padding = 2),
            nn.ReLU()) #output image = 64*33*33
        self.layer1_bn = nn.BatchNorm2d(64)
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 4, stride = 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)) #output image = 64*9*9
        
        self.drop_out = nn.Dropout()
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 4, stride = 1, padding = 2),
            nn.ReLU())
        self.layer3_bn = nn.BatchNorm2d(64)
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 4, stride = 1, padding = 2),
            nn.ReLU())
        self.layer4_bn = nn.BatchNorm2d(64)
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 4, stride = 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.drop_out2 = nn.Dropout()
        
        self.linear1 = nn.Linear(10*10*64, 500)
        self.linear2 = nn.Linear(500, 10)
        
    def forward(self, x):
        x = self.layer1(x)
        #print(x.size())
        
        x = self.layer1_bn(x)
        #print(x.size())
        
        x = self.layer2(x)
        #print(x.size())
        
        x = self.drop_out(x)
        
        x = self.layer3(x)
        
        x = self.layer3_bn(x)
        
        x = self.layer4(x)
        
        x = self.layer4_bn(x)
        
        x = self.layer5(x)
        
        x = self.drop_out2(x)
        
        x = x.view(x.size(0), -1)
        #print(x.size())
        
        x = self.linear1(x)
        #print(x.size())
        
        x = self.linear2(x)
        #print(x.size())
        
        #return nn.Softmax(x)
        return(x)


# In[215]:

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), 
                                lr = learn_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                            step_size = scheduler_step_size, 
                                            gamma = scheduler_gamma)


# In[216]:


for epochs in range(num_epochs):
    scheduler.step()
    correct = 0
    total = 0
    print('Current epoch: ', epochs+1, '/', num_epochs)
    for images, labels in train_loader:
        #images = images.reshape(-1, 16*16)
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == labels).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_acc = correct/total
    print('Training accuracy: ', train_acc)
    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
        
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
    test_acc = correct/total
    print('Test Accuracy: ', test_acc)
