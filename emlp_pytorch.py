import torch
import emlp.nn.pytorch as nn
from emlp.groups import SO13
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from emlp.datasets import ParticleInteraction
import matplotlib.pyplot as plt

trainset = ParticleInteraction(300) # Initialize dataset with 1000 examples
testset = ParticleInteraction(1000)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BS=500
lr=3e-3
NUM_EPOCHS=500

model = nn.EMLP(trainset.rep_in,trainset.rep_out,group=SO13(),num_layers=3,ch=384).to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)

def loss(x, y):
    yhat = model(x.to(device))
    return ((yhat-y.to(device))**2).mean()

def train_op(x, y):
    optimizer.zero_grad()
    lossval = loss(x,y)
    lossval.backward()
    optimizer.step()
    return lossval

trainloader = DataLoader(trainset,batch_size=BS,shuffle=True)
testloader = DataLoader(testset,batch_size=BS,shuffle=True)

test_losses = []
train_losses = []
for epoch in tqdm(range(NUM_EPOCHS)):
    train_losses.append(np.mean([train_op(*mb).cpu().data.numpy() for mb in trainloader]))
    if not epoch%10:
        with torch.no_grad():
            test_losses.append(np.mean([loss(*mb).cpu().data.numpy() for mb in testloader]))


plt.plot(np.arange(NUM_EPOCHS),train_losses,label='Train loss')
plt.plot(np.arange(0,NUM_EPOCHS,10),test_losses,label='Test loss')
plt.legend()
plt.yscale('log')