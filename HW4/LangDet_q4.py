#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 09:53:32 2023

@author: nuohaoliu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from collections import Counter 
import pprint

path = './languageID/'
name1 = 'e0.txt'


def count(language, No_document):
    name1 = str(language)+'10.txt'
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

CondCount = count('e',0)

# print(value)


# {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
# # print(x)