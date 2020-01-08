# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:28:57 2019

@author: Kepeng Qiu

"""


import visualize
from Svdd import Svdd
import scipy.io as sio
import matplotlib.pyplot as plt

# load data
data = sio.loadmat(".\\data\\"+"data.mat")
X = data['X'] # training data 
Y = data['Y'] # testng data


## set kernel parameters
ker = {"type": 'gauss', "width": 5}
#ker = {"type": 'linear', "offset": 0}
#ker = {"type": 'ploy', "degree": 2, "offset": 0}
#ker = {"type": 'tanh', "gamma": 0.01, "offset": 5}
#ker = {"type": 'exp', "width": 5}
#ker = {"type": 'lapl', "width": 12}



# set SVDD parameters
C = 0.7

svdd = Svdd(C, ker)
result = svdd.fit(X,Y,"SPE")

plt.show()
