#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 09:53:32 2023

@author: nuohaoliu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics import confusion_matrix,classification_report

from collections import Counter 
import pprint

path = './languageID/'
name1 = 'e0.txt'

def count(language, No_document):
    name1 = str(language)+ str(No_document) + '.txt'
    with open(path + name1, 'r') as info:
      count = Counter(info.read().lower())
      
    
              
    characters = [' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n',
                  'o','p','q','r','s','t','u','v','w','x','y','z']
    CondCount = np.empty((0,2))
    
    # CondProb = [[],[]]
    i = 0.0
    total = sum(count.values()) - count['\n']
    
    
    for char in characters:
        # list1 = [char,count[char]]
        # # print(np.shape(list1))
        # CondProb.append(list1)
        if (char in count):
            pos = np.reshape(np.array([i,count[char]]),(1,2))
        else: 
            pos = np.reshape(np.array([i,0]),(1,2))
        CondCount = np.append(CondCount,pos,axis=0)
        i += 1.0
        
    return CondCount



def countprob(language, No_document):
    name1 = str(language)+ str(No_document) + '.txt'
    with open(path + name1, 'r') as info:
      count = Counter(info.read().lower())
      
      for i in range(No_document+1):
          with open(path + str(language)+'{}.txt'.format(i), 'r') as info:
              count1 = Counter(info.read().lower())
              count = dict(Counter(count)+Counter(count1))
              value = pprint.pformat(count)
    
              
    characters = [' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n',
                  'o','p','q','r','s','t','u','v','w','x','y','z']
    CondProb = np.empty((0,2))
    
    # CondProb = [[],[]]
    i = 0.0
    total = sum(count.values()) - count['\n']
    
    
    for char in characters:
        # list1 = [char,count[char]]
        # # print(np.shape(list1))
        # CondProb.append(list1)
        if (char in count):
            pos = np.reshape(np.array([i,(count[char]+0.5)/(total+13.5)]),(1,2))
        else: 
            pos = np.reshape(np.array([i,0.5/13.5]),(1,2))
        CondProb = np.append(CondProb,pos,axis=0)
        i += 1.0
        
    return CondProb

CondProbe = countprob('e',9)
CondProbj = countprob('j',9)
CondProbs = countprob('s',9)


def multi(condprob, condcount):
    MultiNominal = np.empty(0)
    prob = condprob[:,1]
    count = condcount[:,1]
    for i in range(len(condprob)): 
    # MultiNominal = np.append(MultiNominal,math.log(prob[i]**condcount[i]))
        # print(prob[i],count[i],math.log(prob[i]**count[i]))
        MultiNominal = np.append(MultiNominal,count[i]*math.log(prob[i]))
    return sum(MultiNominal)

Multi_e = np.empty(0)
Multi_j = np.empty(0)
Multi_s = np.empty(0)

for i in range(10,20,1):
    CondCounte = count('e',i)
    CondCountj = count('e',i)
    CondCounts = count('e',i)
    
    Multi_e = np.append(Multi_e,multi(CondProbe, CondCounte))
    Multi_j = np.append(Multi_j,multi(CondProbj, CondCountj))
    Multi_s = np.append(Multi_s,multi(CondProbs, CondCounts))
 #%%   
Multi_e = np.reshape(Multi_e,(10,1))
Multi_j = np.reshape(Multi_j,(10,1))
Multi_s = np.reshape(Multi_s,(10,1))
Multi = np.concatenate((Multi_e, Multi_j,Multi_s),axis=1)
# print(value)


def shuffle(name):
    with open(path + str(name) + ".txt", mode="r", encoding="utf-8") as myFile:
        lines = list(myFile)
    random.shuffle(lines)
    return lines

lines = shuffle('e10')

# {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
# # print(x)