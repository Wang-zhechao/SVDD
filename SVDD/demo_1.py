# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:37:51 2019

@author: Kepeng Qiu

"""
import numpy as np
import visualize
from Svdd import Svdd
import pandas as pd
import matplotlib.pyplot as plt

# load data
irisData = pd.read_csv(".\\SVDD\\SVDD\\data\\"+"SPE_teX_history.csv", header=0)
# irisData = pd.read_csv(".\\SVDD\\SVDD\\data\\"+"H2_teX_history.csv", header=0)
data = np.array(irisData.iloc[:, 0:1], dtype=float)

# X: training data; Y: testng data
X = data [0:700, :]
Y = data [0:1050, :]

# set kernel parameters
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
# # train SVDD model
# model = svdd.train(X)
# # test SVDD model
# distance = svdd.test(model, Y)
# # plot the testing results
# svdd.plotTestResult(model, distance, "SPE")
plt.show()