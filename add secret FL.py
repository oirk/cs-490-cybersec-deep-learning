# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 19:14:31 2019

@author: oirk
"""

import torch as th
from torch import nn, optim
import random

import syft as sy

hook = sy.TorchHook(th)  

Q = 23740629843760239486723

# accepts a secrete and shares it among n_shahre, returns a tuple with n_share shares
def encrypt(x, n_share=3):
    
    shares = list()
    
    for i in range(n_share-1):
        shares.append(random.randint(0,Q))
        
    shares.append(Q - (sum(shares) % Q) + x)
    
    return tuple(shares)

  # accepts a tuple of shares and return the decrypted values
def decrypt(shares):
    return sum(shares) % Q

def add(a, b):
    c = list()
    assert(len(a) == len(b))
    for i in range(len(a)):
        c.append((a[i] + b[i]) % Q)
    return tuple(c)

bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
secureWorker =sy.VirtualWorker(hook, id="secureWorker")

x = th.tensor([1,2,3])
x = x.share(bob,alice,secureWorker)

model = nn.Linear(2,1)

model = model.share(bob,alice)