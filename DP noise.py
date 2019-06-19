# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:35:20 2019

@author: oirk
"""
import random
import torch
import statistics

def random_guess(db,index):
    guess_list = [0,1]
    guess_pref = random.random(guess_list)
    coin = random.randrange(1,3)
    if (coin == 1) :
        db[index].fill_(guess_pref)
        return db
    else:
        coin = random.randrange(1,3)
        if (coin == 1) :
            db[index].fill_(1)
            return db
        else:
            db[index].fill_(0)
            return db

#add noise
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
        


#create DB
def create_db_static(entries):
  return torch.FloatTensor( torch.rand(entries) ) >.5

#.5+truth +(1-noise)(.5)
    
#truth= 2(ans-0.25)

#need query 
def query_mean (db):
    return db.mean()
    

#def find_truth(db):
    
dbs =(create_db_static(1000)).type(torch.FloatTensor)




new_list = []
for item in dbs:
    temp = random_switch(item.item())
    new_list.append(temp)


mean_dbs = dbs.mean()
mean_noise = statistics.mean(new_list)
print ()
print ("this is normal -->",mean_dbs)
print ()
print ("this is with noise-->",mean_noise)

truth= 2*((mean_noise)-0.25)

print ("this is truth --> ",truth)