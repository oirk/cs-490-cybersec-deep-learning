# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:58:09 2019

@author: oirk
"""

import torch

num_entries = 5000

def create_db(entries):
  return torch.FloatTensor(torch.rand(entries)  )



# create a function to create paraller db
def get_parallel_dbs(db):

    parallel_dbs = list()

    for i in range(len(db)):
        pdb = torch.cat((db[0:i], db[i+1:]))
        parallel_dbs.append(pdb)
    #print(f'A databse of size {db.shape} was created')
    return parallel_dbs

def get_parallel_ten(db):

    parallel_dbs = list()

    for i in range(len(db)):
        pdb = torch.cat((db[0:i], db[i+1:]))
        parallel_dbs.append(pdb)
    #print(f'A databse of size {db.shape} was created')
    return parallel_dbs

def query (db):
    return db.sum()

#helper 
def get_db_and_parallel(num_entries):
    db = torch.rand(num_entries)
    pdb = get_parallel_dbs(db)
    return db, pdb

db,pdbs = get_db_and_parallel(num_entries)


def query_mean (db):
    return db.mean()
    

def sensitivity(query, num_items):
    
  
  # generate the db and the pdbs
    db = create_db(num_items)
    pdbs = get_parallel_dbs(db)
  # query the db
    db_resualt = query(db)
  
  # return the value of sensitivity
  # loop on each pdb in the pdbs and calculate its distance
  # find the maximum distance <-- that's your sensitivity
    dif_list = []
    for item in pdbs:
        pbds_res =  query(item)
        dif_list.append(pbds_res)
    sens_list = []   
    for item in dif_list : 
        sens_list.append(db_resualt - item)

    sensitivit  = max(sens_list)
    
    return sensitivit

print (sensitivity(query,5000))


def check_sum(db,threshold):
    if (db.sum() >= threshold):
        return float(True)
    else:
        return float(False) 
    
new_db = create_db(10)
print (check_sum(new_db,5))



def sum_attack(db,pdbs,person):
    
    return (db.sum() - sum(pdbs[person+1]))

db,pdbs = get_db_and_parallel(1000)

sum_person = sum_attack(db,pdbs,10)
print (sum_person)


def mean_attack(db,pdbs,person):
    
    return (db.mean() * (pdbs[person+1]).mean())

mean_person = mean_attack(db,pdbs,10)
print (mean_person)


print ((sum(db)>49) - (sum(pdbs[11])>49))







