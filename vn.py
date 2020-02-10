# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:14:42 2019

@author: farih
"""

import math
import numpy as np
import numpy as geek


#1-D


mu = 0
cov = 1
x = np.random.normal(mu, cov, 1000)  
a1=(x[0])
a2=(x[1])
distance=math.sqrt((a1-a2)**2)
print(distance)



#2-d

mean = [0, 0]
cov = [[1, 0], [0, 1]]
x, y = np.random.multivariate_normal(mean, cov, 1000).T
a1=(x[0],y[0])
a2=(x[1],y[1])
distancen = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a1, a2)]))
print(distancen)



# 4-D
mu = [0,0,0,0]
cov = geek.identity(4) 
b1,b2,b3,b4 = np.random.multivariate_normal(mu, cov, 1000).T

a1=(b1[0],b2[0],b3[0],b4[0])
a2=(b1[1],b2[1],b3[1],b4[1])
distancen = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a1, a2)]))
print(distancen)

# 8-D

mu = [0,0,0,0,0,0,0,0]
cov = geek.identity(8) 
b1,b2,b3,b4,b5,b6,b7,b8 = np.random.multivariate_normal(mu, cov, 1000).T
a1=(b1[0],b2[0],b3[0],b4[0],b5[0],b6[0],b7[0],b8[0])
a2=(b1[1],b2[1],b3[1],b4[1],b5[1],b6[1],b7[1],b8[1])

distancen = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a1, a2)]))
print(distancen)



