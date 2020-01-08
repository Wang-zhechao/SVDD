# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:33:23 2019

@author: Kepeng Qiu
"""

import numpy as np
import matplotlib.pyplot as plt

class Svdd():
    '''
    C         Trade-off parameter\n
    ker       Kernel function parameters\n 
    ker = {"type": 'gauss', "width": s}\n
    ker = {"type": 'linear', "offset": c}\n
    ker = {"type": 'ploy', "degree": d, "offset": c}\n
    ker = {"type": 'tanh', "gamma": g, "offset": c}\n
    ker = {"type": 'exp', "width": s}\n
    ker = {"type": 'lapl', "width": s} 
    
    '''
    def __init__(self, C = 0.8, ker = {"type": 'gauss', "width": 2}):
        
        ''' 
        DESCRIPTION
        
        --------------------------------------------------        
        INPUT
         C         Trade-off parameter
         ker       Kernel function parameters              

        '''                
        self.C = C
        self.ker = ker


    def train(self, X):
        
        ''' 
        DESCRIPTION
        
        Train SVDD model
        
        -------------------------------------------------- 
        Reference
        Tax, David MJ, and Robert PW Duin.
        "Support vector data description." 
        Machine learning 54.1 (2004): 45-66.
        
        -------------------------------------------------- 
        model = train(X)
        
        --------------------------------------------------        
        INPUT
        X         Training data
                        
        OUTPUT
        model     SVDD hypersphere
        --------------------------------------------------
        
        '''
        
        threshold = 1e-10
        # compute the kernel matrix
        K = self.getMatrix(X)
        
        # solve the Lagrange dual problem using the SMO algorithm
        alf = self.smo(K)
    
        # support vector
        tmp_1 = np.mat(np.where(alf > threshold))
        SV_index = (tmp_1[0, :]).T
        SV_value = X[SV_index, :]
        SV_alf = alf[SV_index, 0]
        
        # compute the center: eq(7)
        center = np.dot(alf.T, X)
        
        # Compute the radius: eq(15)
        # The distance from any support vector to the center of 
        # the sphere is the hypersphere radius. 
        # Here take the 1st support vector as an example.
        
        r_index = SV_index[0, 0];
        # the 1st term in eq(15)
        term_1 = K[r_index, r_index]
        # the 2nd term in eq(15)
        term_2 = -2*np.dot(K[r_index, :], alf)
        
        # the 3rd term in eq(15)
        tmp_2 = np.mat(np.dot(alf, alf.T)*K)
        term_3 = (tmp_2.sum(axis = 0)).sum(axis = 1)
    
        # radius
        radius = term_1+term_2+term_3;
        
        # Store the results
        model = {"X"         : X         ,
                 "SV_alf"    : SV_alf    ,
                 "radius"    : radius,
                 "SV_value"  : SV_value  ,
                 "SV_index"  : SV_index  ,
                 "center"    : center    ,
                 "term_3"    : term_3    ,
                 "alf"       : alf       ,  
                 "K"         : K         ,
                 "ker"       : self.ker  ,
                 "C"         : self.C    ,
                 }
        
        return model

    
    def test(self, model, Y):
    
        ''' 
        DESCRIPTION
        
        Test the testing data using the SVDD model
    
        distance = test(model, Y)
        
        --------------------------------------------------        
        INPUT
        model     SVDD hypersphere            
        Y         Testing data
            
        OUTPUT
        distance  Distance between the testing data and hypersphere
        --------------------------------------------------
        
        '''    
        
        
        n = Y.shape[0]
        
        # compute the kernel matrix
        K = self.getMatrix(Y, model["X"])
        
        # the 1st term
        term_1 = self.getMatrix(Y)
        
        # the 2nd term
        tmp_1 = -2*np.dot(K, model["alf"])
        term_2 = np.tile(tmp_1, (1, n))
        
        # the 3rd term
        term_3 =  model["term_3"]
        # distance
        d = np.diagonal(term_1+term_2+term_3) 
        
        return d 

    def smo(self, K):
    
        '''
        DESCRIPTION
        
        Solve the Lagrange dual problem
        
        alf = smo(K)
        
        --------------------------------------------------
        INPUT
        K         Kernel matrix
                        
        OUTPUT
        alf       Lagrange multipliers
        --------------------------------------------------
        '''   

        # initialize 'alf'
        n = K.shape[0]
        alf = np.ones((n, 1))/n
        
        # initialize 'g'
        # g(i) is the partial derivative of the objective function 
        # of the dual problem to alf(i)
        g = np.ones((n, 1))
        g[:, 0] = np.diagonal(K) 
        tmp = np.ones((n, n))/n
        tmp1 = tmp*K
        tmp2 = np.mat(tmp1.sum(axis = 0))
        g = g-tmp2.T-alf*g
        
        #
        gmax = np.max(g)
        i = np.argmax(g)  
        gmin = np.min(g)
        j = np.argmin(g)
        delta = gmax-gmin
        
        #
        tor = 1e-5
        iteration = 0
        max_iter = 1000
        eps = 1e-10
        
        for iIteration in range(0, max_iter):
            
            # compute the step
            C1 = self.C-alf[i, 0]
            C2 = alf[j, 0]
            C3 = (g[i, 0]-g[j, 0])/(K[i, i]+K[j, j]-2*K[i, j])
            L = np.min([C1, C2, C3])
                
            # update the partial derivative
            g = g-(np.mat(L*K[:, i])).T+(np.mat(L*K[:, j])).T
            
            # update the Lagrange multipliers
            alf[i, 0] = alf[i, 0]+L
            alf[j, 0] = alf[j, 0]-L
            
            # choose the working set
            IDi_tmp = np.mat(np.where(alf < self.C-eps))
            IDi = (IDi_tmp[0, :]).T
            
            IDj_tmp = np.mat(np.where(alf > eps))
            IDj = (IDj_tmp[0, :]).T
            
            gmax = np.max(g[IDi, 0])
            IDix = np.argmax(g[IDi, 0])
            
            gmin = np.min(g[IDj, 0])
            IDjx = np.argmin(g[IDj, 0])
            
            i = IDi[IDix, 0]
            j = IDj[IDjx, 0]
            
            # compute the error
            iteration = iteration+1
            delta = gmax-gmin
            if delta < tor:
                break
        return alf

    def getMatrix(self, X, *args):
    
        ''' 
        DESCRIPTION
        
        Compute kernel matrix 
        
        K = getMatrix(X, *Y)
        
        -------------------------------------------------- 
        INPUT
        X         Training data

        OUTPUT
        K         Kernel matrix 
        -------------------------------------------------- 
                        
                            
        type   -  
        
        linear :  k(x,y) = x'*y+c
        poly   :  k(x,y) = (x'*y+c)^d
        gauss  :  k(x,y) = exp(-||x-y||^2/(2*s^2))
        tanh   :  k(x,y) = tanh(g*x'*y+c)
        exp    :  k(x,y) = exp(-||x-y||/(2*s^2))
        lapl   :  k(x,y) = exp(--||x-y||/(2*s))
           
        degree -  d
        offset -  c
        width  -  s
        gamma  -  g
        
        --------------------------------------------------      
        ker    - 
        
        ker = {"type": 'gauss', "width": s}
        ker = {"type": 'linear', "offset": c}
        ker = {"type": 'ploy', "degree": d, "offset": c}
        ker = {"type": 'tanh', "gamma": g, "offset": c}
        ker = {"type": 'exp', "width": s}
        ker = {"type": 'lapl', "width": s}
        '''
        def getPairDistances(X, Y):
            
            YT = Y.transpose()
            vecProd = np.dot(X, YT)
            SqX =  X**2
            sumSqX = np.matrix(np.sum(SqX, axis = 1))
            sumSqAEx = np.tile(sumSqX.transpose(), (1, vecProd.shape[1]))
        
            SqY = Y**2
            sumSqY = np.sum(SqY, axis = 1)
            sumSqBEx = np.tile(sumSqY, (vecProd.shape[0], 1))
            PairDistances = sumSqBEx + sumSqAEx - 2*vecProd
            PairDistances[PairDistances<0] = 0
            
            return PairDistances

        
        def gaussFunc():
            
            if self.ker.__contains__("width"):
                s =  self.ker["width"]
            else:
                s = 2
            
            if mode == "train":           
                tmp = np.sum(X**2, axis = -1)
                K = np.exp(-(tmp[:,None]+tmp[None, :]-2*np.dot(X, X.T))/(2*s**2))
            
            if mode == "test":           
                tmp = getPairDistances(X, Y)
                K = np.exp(-tmp/(2*s**2))
                
            return K
            
        def linearFunc():
            
            if self.ker.__contains__("offset"):
                c =  self.ker["offset"]
            else:
                c = 0

            K = np.dot(X, Y.T)+c
            
            return K
        
        def ployFunc():
            if self.ker.__contains__("degree"):
                d =  self.ker["degree"]
            else:
                d = 2
                
            if self.ker.__contains__("offset"):
                c =  self.ker["offset"]
            else:
                c = 0
                
            K = (np.dot(X, Y.T))**d+c
            
            return K
        
        def expFunc():
            
            if self.ker.__contains__("width"):
                s =  self.ker["width"]
            else:
                s = 2
            
            if mode == "train":           
                tmp_1 = np.sum(X**2, axis = -1)
                tmp_2 = tmp_1[:,None]+tmp_1[None, :]-2*np.dot(X, X.T)
                tmp_2[tmp_2<eps] = 0
                K = np.exp(-np.sqrt(tmp_2)/(2*s**2))
            
            if mode == "test":           
                tmp = np.sqrt(getPairDistances(X, Y))
                K = np.exp(-tmp/(2*s**2))
                
            return K
        
        def laplFunc():
            
            if self.ker.__contains__("width"):
                s =  self.ker["width"]
            else:
                s = 2
            
            if mode == "train":           
                tmp_1 = np.sum(X**2, axis = -1)
                tmp_2 = tmp_1[:,None]+tmp_1[None, :]-2*np.dot(X, X.T)
                tmp_2[tmp_2<eps] = 0
                K = np.exp(-np.sqrt(tmp_2)/(2*s))
            
            if mode == "test":           
                tmp = np.sqrt(getPairDistances(X, Y))
                K = np.exp(-tmp/(2*s))
                
            return K
    
        def tanhFunc():
            if self.ker.__contains__("gamma"):
                g =  self.ker["gamma"]
            else:
                g = 0.01
                
            if self.ker.__contains__("offset"):
                c =  self.ker["offset"]
            else:
                c = 0
            
            tmp = g*np.dot(X, Y.T)+c
            K = (np.exp(tmp)-np.exp(-tmp))/(np.exp(tmp)+np.exp(-tmp))

            return K

        kernelType = self.ker["type"]
        eps = 1e-10
        
        if len(args) == 0:
            mode = "train"
            Y = X
         
        if len(args) == 1:
            mode = "test"
            Y = args[0]         
   
        switcher = {    
                        "gauss"   : gaussFunc  ,        
                        "linear"  : linearFunc ,
                        "ploy"    : ployFunc   ,
                        "exp"     : expFunc    ,
                        "lapl"    : laplFunc   ,
                        "tanh"    : tanhFunc   ,
                     }
        
        return switcher[kernelType]()


    def plotTestResult(self, model, distance, title):

        ''' 
        DESCRIPTION
        
        Plot the testing results
        
        plotTestResult(model, distance, title)
        
        -------------------------------------------------- 
        INPUT
        model     SVDD hypersphere
        distance 
        title     figure's 
        -------------------------------------------------- 
        '''
        fontSize = 18

        n = distance.shape[0]
        plt.figure(figsize = (8, 6))
        ax = plt.subplot(1, 1, 1)
        radius = np.ones((n, 1))*model["radius"]
        ax.plot(radius, 
                color ='r',
                linestyle = '-', 
                marker = 'None',
                linewidth = 1, 
                markeredgecolor ='k',
                markerfacecolor = 'w', 
                markersize = 6)
        
        ax.plot(distance,
                color = 'k',
                linestyle = '-',
                marker='None',
                linewidth=1,
                markeredgecolor = 'k',
                markerfacecolor = 'C4',
                markersize = 6)
        
        ax.set_xlabel('Samples', {'size': fontSize*1.1})
        ax.set_ylabel('Distance', {'size': fontSize*1.1})
        
        ax.legend(["Radius","Distance"], 
                ncol = 1, loc = 0,
                prop = {'size': fontSize*0.9}, 
                edgecolor = 'black', 
                markerscale = 2, fancybox = True)
        
        ax.tick_params(labelsize = fontSize)
        ax.set_title(title, {'size': fontSize*1.1})

    def fit(self, traindata, testdata, plottitle):

        ''' 
        DESCRIPTION
        
        Plot the testing results
        
        fit(traindata, testdata, plottitle)
        
        -------------------------------------------------- 
        INPUT\n
        traindata     SVDD hypersphere
        testdata 
        plottitle     figure's title
        -------------------------------------------------- 
        OUTPUT\n
        model = {"X"         : traindata ,
                 "SV_alf"    : SV_alf    ,
                 "radius"    : radius    ,
                 "SV_value"  : SV_value  ,
                 "SV_index"  : SV_index  ,
                 "center"    : center    ,
                 "term_3"    : term_3    ,
                 "alf"       : alf       ,  
                 "K"         : K         ,
                 "ker"       : self.ker  ,
                 "C"         : self.C    ,
                 }        
        --------------------------------------------------
        '''

        model = self.train(traindata)
        # test SVDD model
        distance = self.test(model, testdata)
        # plot the testing results
        self.plotTestResult(model, distance, plottitle)
        return model


    
    