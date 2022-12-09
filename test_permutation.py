import torch
import emlp.nn.pytorch as nn
import torch
import emlp.nn.pytorch as nn


from emlp.reps import T,V
from emlp.groups import SO13, Z, S

repin= 4*V # Setup some example data representations
repout = V**0
G = Z(4) # The lorentz group

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.randn(5,repin(G).size()).to(device) # generate some random data

model = nn.EMLP(repin,repout,G).to(device) # initialize the model

model(x)