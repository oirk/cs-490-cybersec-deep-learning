# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 22:21:59 2019

@author: oirk
"""

import random
import torch
import statistics

def random_switch(item):
    temp = item
    guess_list=[0,1]
    coin = random.choice(guess_list)
    
    if (coin == 1) :
        return item
    else:
        coin2 = random.choice(guess_list)
       
        if coin2 == 1:
            temp=1
            return (temp)
        else:
            temp =0
            return (temp)
        

def random_switch_bias(item):
    temp = item
    guess_list=[0,1,2,3,4,5,6,7,8,9]
    coin = random.choice(guess_list)
    
    if (coin == 0 or coin == 1 or coin == 3 or coin == 4 or coin == 5 or coin == 6 or coin == 7 ) :
        return item
    else:
        coin2 = random.choice(guess_list)
       
        if (coin2 == 0 or coin2 == 1 or coin2 == 3 or coin2 == 4 or coin2 == 5 or coin2 == 6 or coin2 == 7 ):
            temp=1
            return (temp)
        else:
            temp =0
            return (temp)
        
def create_db_static(entries):
  return torch.FloatTensor( torch.rand(entries) ) >.5

dbs =(create_db_static(100)).type(torch.FloatTensor)

def get_parallel_dbs(db):

    parallel_dbs = list()

    for i in range(len(db)):
        pdb = torch.cat((db[0:i], db[i+1:]))
        parallel_dbs.append(pdb)
    #print(f'A databse of size {db.shape} was created')
    return parallel_dbs

def sensitivity(query):
  # return the value of sensitivity
  # loop on each pdb in the pdbs and calculate its distance
  # find the maximum distance <-- that's your sensitivity
    dif_list = []
    for item in pdbs:
        pbds_res =  query(item)
        dif_list.append(pbds_res)
    sens_list = []   
    for item in dif_list : 
        sens_list.append(query- item)

    sensitivit  = max(sens_list)
    
    return sensitivit

pdbs = get_parallel_dbs(dbs)


def query(db):
    return torch.sum(db.float())
def M(db):
    noise = b = sensitivity(query) / epsilon
    query(db) + noise