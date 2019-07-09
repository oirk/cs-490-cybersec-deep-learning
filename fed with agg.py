# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 20:01:54 2019

@author: oirk
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:59:58 2019

@author: oirk
"""
import torch as th
from torch import nn, optim


import syft as sy

hook = sy.TorchHook(th)  


#start of exercises 
# create workers
bob = sy.VirtualWorker(hook, id="bo")
alice = sy.VirtualWorker(hook, id="alic")
secureWorker =sy.VirtualWorker(hook, id="secureWorke")
compute_nodes = [bob, alice]
# Make each worker aware of the other workers


# A Toy Dataset
data = th.tensor([[1.,1],[0,1],[1,0],[0,0]], requires_grad=True)
target = th.tensor([[1.],[1], [0], [0]], requires_grad=True)

# create local datasets at Bob and Alice

data_bob = data.send(bob)
target_bob = target.send(bob)


data_alice = data.send(alice)
target_alice = target.send(alice)



# create a linear model at local worker
model = nn.Linear(2,1)

opt = optim.SGD(params=model.parameters(),lr=0.1)

# Send copies of the linear model to alice and bob
bobs_model = model.copy().send(bob)
alices_model = model.copy().send(alice)

bobs_opt = optim.SGD(params=bobs_model.parameters(),lr=0.1)
alices_opt = optim.SGD(params=alices_model.parameters(),lr=0.1)

# train the models at alice and bob
# move the models to the third worker
# aggreegate the models
# send the model back to the local worker
for i in range(10):

    # Train Bob's Model
    bobs_opt.zero_grad()
    bobs_pred = bobs_model(data_bob)
    bobs_loss = ((bobs_pred - target_bob)**2).sum()
    bobs_loss.backward()

    bobs_opt.step()
    bobs_loss = bobs_loss.get().data

    # Train Alice's Model
    alices_opt.zero_grad()
    alices_pred = alices_model(data_alice)
    alices_loss = ((alices_pred - target_alice)**2).sum()
    alices_loss.backward()

    alices_opt.step()
    alices_loss = alices_loss.get().data
    
   

alices_model.move(secureWorker)
bobs_model.move(secureWorker)
  
with th.no_grad():
    model.weight.set_(((alices_model.weight.data + bobs_model.weight.data) / 2).get())
    model.bias.set_(((alices_model.bias.data + bobs_model.bias.data) / 2).get())
""
print("Bob:" + str(bobs_loss) + " Alice:" + str(alices_loss))

bobs_model = model.copy().send(bob)
alices_model = model.copy().send(alice)

