# -*- coding: utf-8 -*-
import numpy as np
import json
from math import gamma
from scipy.interpolate import CubicSpline

def elu(x):
# Implements the elu activation function.
    val = x
    val[x < 0] = np.exp(x[x < 0]) - 1
    return(val)

class NeuralNetwork:    
# Implements a basic neural network.
    def __init__(self,fileName):
    # Constructor, network is loaded from a json file.
        self.weights = []
        self.biases = []
        self.scaleMeanIn = []
        self.scaleStdIn = []
        self.scaleMeanOut = []
        self.scaleStdOut = []
        with open(fileName) as json_file:
            tmp = json.load(json_file)
        nLayers = int((len(tmp)-4)/2)
        for i in range(nLayers):
            self.weights.append(np.transpose(tmp[2*i]))
            self.biases.append(np.array(tmp[2*i+1]).reshape(-1,1))
        self.scaleMeanIn = np.array(tmp[-4]).reshape(-1,1)
        self.scaleStdIn = np.sqrt(np.array(tmp[-3]).reshape(-1,1))
        self.scaleMeanOut = np.array(tmp[-2]).reshape(-1,1)
        self.scaleStdOut = np.sqrt(np.array(tmp[-1]).reshape(-1,1))
        
    def Eval(self,x):
    # Evaluates the network.
        nLayers = len(self.weights)
        val = (x - self.scaleMeanIn)/self.scaleStdIn
        for i in range(0,nLayers - 1):
            val = elu(np.dot(self.weights[i],val) + self.biases[i])
        val = np.dot(self.weights[nLayers-1],val) + self.biases[nLayers-1]
        return self.scaleStdOut*val + self.scaleMeanOut
    
class NeuralNetworkPricer:    
# Implements a neural network pricer based on multiple sub networks.
    def __init__(self,contracts_folder,weights_folder,model_name):
    # Constructor.
        self.nn = []
        self.idx_in = []
        self.idx_out = []
        self.lb = []
        self.ub = []
        self.label = model_name
        
        # Load contract grid:
        self.T = np.loadtxt(contracts_folder + "\\expiries.txt").reshape(-1,1)
        self.k = np.loadtxt(contracts_folder + "\\logMoneyness.txt").reshape(-1,1)
        
        # Set the forward variance curve grid points (in case relevant):
        Txi = np.array([0.0025,0.0050,0.0075,0.0100,0.0125,0.0150,0.0175,0.0200,
                        0.0400,0.0600,0.0800,0.1000,0.1200,0.1400,0.1600,0.2800,
                        0.4000,0.5200,0.6400,0.7600,0.8800,1.0000,1.2500,1.5000,
                        1.7500,2.0000,3.0000])
        
        # Basic naming of json files:
        json_files = ["_weights_1.json",
                      "_weights_2.json",
                      "_weights_3.json",
                      "_weights_4.json",
                      "_weights_5.json",
                      "_weights_6.json"]

        # Load each sub-network:
        idxOutStart = 0
        for i in range(len(json_files)):
            self.nn.append(NeuralNetwork(weights_folder + "\\" + model_name + json_files[i]))
            self.idx_in.append(np.arange(0,self.nn[i].scaleMeanIn.shape[0]))
            idxOutEnd = idxOutStart + self.nn[i].scaleMeanOut.shape[0]
            self.idx_out.append(np.arange(idxOutStart,idxOutEnd))
            idxOutStart = idxOutEnd
        
        # Set parameter bounds (and more):
        if (model_name == "rheston"):
            self.lb = np.concatenate((np.array([0,0.1,-1]),pow(0.05,2)*np.ones(28))).reshape(-1,1)
            self.ub = np.concatenate((np.array([0.5,1.25,0]),np.ones(28))).reshape(-1,1)
            self.Txi = np.concatenate((np.array([0]),Txi))
        elif (model_name == "rbergomi"):
            self.lb = np.concatenate((np.array([0,0.75,-1]),pow(0.05,2)*np.ones(27))).reshape(-1,1)
            self.ub = np.concatenate((np.array([0.5,3.50,0]),np.ones(27))).reshape(-1,1)
            self.Txi = Txi
        elif (model_name == "rbergomi_extended"):
            self.lb = np.concatenate((np.array([0.75,-1,-0.5,-0.5]),pow(0.05,2)*np.ones(27))).reshape(-1,1)
            self.ub = np.concatenate((np.array([3.50,0,0.5,0.5]),np.ones(27))).reshape(-1,1)
            self.Txi = Txi
        elif (model_name == "heston"):
            self.lb = np.array([0,pow(0.05,2),0,-1,pow(0.05,2)]).reshape(-1,1)
            self.ub = np.array([25,1,10,0,1]).reshape(-1,1)
        else:
            raise Exception('NeuralNetworkPricer:__init__: Invalid model name.')
        
    def EvalInGrid(self,x):
    # Evaluates the model in the grid points only.
        # Check bounds:
        if (any(x < self.lb) or any(x > self.ub)):
            raise Exception('NeuralNetworkPricer:EvalInGrid: Parameter bounds are violated.')
        
        nNetworks = len(self.nn)
        nPts = self.k.shape[0]
        iv = np.zeros(nPts).reshape(-1,1)
        for i in range(0,nNetworks):
            iv[self.idx_out[i]] = self.nn[i].Eval(x[self.idx_in[i]])
        
        return(iv)
        
    def AreContractsInDomain(self,kq,Tq):
    # Checks if the contracts are within the supported domain.
        if not kq.shape == Tq.shape:
            raise Exception('NeuralNetworkPricer:AreContractsInDomain: Shape of input vectors are not the same.')
    
        uniqT = np.unique(Tq)
        uniqTGrid = np.unique(self.T)
        minTGrid = np.min(uniqTGrid)
        maxTGrid = np.max(uniqTGrid)
        idxValid = np.ones((len(kq), 1), dtype=bool)
        for i in range(0,len(uniqT)):
            idxT = Tq == uniqT[i]
            if uniqT[i] > maxTGrid or uniqT[i] < minTGrid:
                idxValid[idxT] = False
            else:
                if uniqT[i] == maxTGrid:
                    idxAbove = len(uniqTGrid) - 1
                else:
                    idxAbove = np.argmax(uniqTGrid > uniqT[i])
                idxBelow = idxAbove - 1
                idxGridBelow = self.T == uniqTGrid[idxBelow]
                idxGridAbove = self.T == uniqTGrid[idxAbove]
                idxValid[idxT] =   (kq[idxT] >= np.max([np.min(self.k[idxGridBelow]),np.min(self.k[idxGridAbove])])) \
                                 & (kq[idxT] <= np.min([np.max(self.k[idxGridBelow]),np.max(self.k[idxGridAbove])]))
        return(np.ravel(idxValid))        
        
    def Eval(self,x,kq,Tq):
    # Evaluates the model in arbitrary contracts (within the supported domain).
        ivGrid = self.EvalInGrid(x)
        if (not all(self.AreContractsInDomain(kq,Tq))):
            raise Exception('NeuralNetworkPricer:Eval: Some contracts violate the neural network domain.')

        ivGrid = self.EvalInGrid(x)
        nPts = kq.shape[0]
        iv = np.zeros((nPts,1))
        uniqT = np.unique(Tq)
        uniqTGrid = np.unique(self.T)
        maxTGrid = max(uniqTGrid)
        for i in range(0,len(uniqT)):
            idxT = Tq == uniqT[i]
            if uniqT[i] == maxTGrid:
                idxAbove = len(uniqTGrid) - 1
            else:
                idxAbove = np.argmax(uniqTGrid > uniqT[i])
            idxBelow = idxAbove - 1
            T_above = uniqTGrid[idxAbove]
            T_below = uniqTGrid[idxBelow]
            idxGridBelow = self.T == uniqTGrid[idxBelow]
            idxGridAbove = self.T == uniqTGrid[idxAbove]

            iv_below_grid = ivGrid[idxGridBelow]
            iv_above_grid = ivGrid[idxGridAbove]
            k_below_grid = self.k[idxGridBelow]
            k_above_grid = self.k[idxGridAbove]

            # Fit splines:
            idxSort_below = np.argsort(k_below_grid)
            idxSort_above = np.argsort(k_above_grid)
            spline_lower  = CubicSpline(k_below_grid[idxSort_below],iv_below_grid[idxSort_below],bc_type='natural')
            spline_upper  = CubicSpline(k_above_grid[idxSort_above],iv_above_grid[idxSort_above],bc_type='natural')

            # Evaluate spline:
            iv_below = spline_lower(kq[idxT])
            iv_above = spline_upper(kq[idxT])

            # Interpolate
            frac = (uniqT[i] - T_below) / (T_above - T_below)
            iv[idxT] = np.sqrt(((1-frac)*T_below*pow(iv_below,2) + frac*T_above*pow(iv_above,2))/uniqT[i])

        return(iv)
            
def CheckNonNeqReqTheta(v0,H,t,theta):
# Description: In the context of the rough Heston model, we check if 
# theta(t)dt + v0*L(dt) is a non-negative measure where 
# L(dt) = t^(-H-1/2)/gamma(1/2-H) dt.
#
# Parameters:
#   v0:     [1x1 real]  Initial instantaneous variance.
#   H:      [1x1 real]  Hurst exponent.
#   t:      [Nx1 real]  Time points between which theta is piecewise constant 
# 						(zero excluded). We assume theta is extrapolated flat.
#   theta:  [Nx1 real]  Theta values for each interval.
#
# Output:
#   valid:  [1x1 logical] True if theta(t)dt + v0*L(dt) is a non-negative measure.
#   val:    [Nx1 real] The values that should be non-negative.
    
    # Because the density of L(dt) = t^(-H-1/2)/gamma(1/2-H) dt is non-increasing,  
    # it suffices to check the right end points:
    val = theta + (v0/gamma(1/2-H))*pow(t,-H-1/2)

    # For the requirement to be satisfied for t -> infinity also, we need additionally:
    val[-1] = theta[-1]

    # Check if measure is non-negative:
    valid = all(val >= 0)
    
    return [valid,val]
        
def GetThetaFromXi(v0,H,t,xi):
# Description: Consider (in the context of the rough Heston model) the following,
#
#   xi(t) = v0 + int_0^t K(t-s) theta(s) ds,	t >= 0,
#
# where v0 >= 0, K(t) = (1/gamma(H+1/2))*t^(H-1/2), H between 0 and 1/2, and theta 
# is a deterministic function that is piecewise constant betwen the time points
# 0 < t_1 < t_2 < ... < t_n.
#
# Given v0 and xi(t_i), i=1,2,...,n, we return the theta-values that ensures 
# that t -> (t,xi(t)) goes through the points (t_i,xi(t_i)), i=1,2,...,n.
#
# Parameters:
# 	v0:		[1x1 real] Instantaneous variance.
#   H:      [1x1 real] Hurst exponent.
#   t:      [nx1 real] Time points t_1,t_2,...,t_n.
#   xi:     [nx1 real] Forward variances at the maturities t_1,t_2,...,t_n.
#   
# Output:
#   theta:  [nx1 real] Theta values.
#

    t_ext = np.concatenate((np.zeros((1,1)),t))
    n = len(xi);
    theta = np.zeros((n,1))
    for i in range(0,n):
        if i==0:
            wii = (1/gamma(H+3/2))*pow(t[i],H+1/2)
            wik_theta_sum = 0
        else:
            wik_theta_sum = 0
            for j in range(1,i+1):
                wik_theta_sum = wik_theta_sum + (1/gamma(H+3/2))*(pow(t[i]-t_ext[j-1],H+1/2) 
                                                                - pow(t[i]-t_ext[j],H+1/2))*theta[j-1]
            wii = (1/gamma(H+3/2))*pow(t[i]-t[i-1],H+1/2)
        theta[i] = (xi[i] - v0 - wik_theta_sum) / wii

    return theta            
            