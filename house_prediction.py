#I prefer to work in Pandas dataFrame to extract and label data
#Then convert to array formats
#PREDICTION OF MEDIAN HOUSE PRICES IN THE US AROUND THE TIME OF THE 2008 CRISIS
#DIFFERENT ALGORITHMS FOR AN ACCURATE PREDICTION.
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

D = pd.read_csv('GlobalHousePriceIndex.csv',',')
t = D['dateq']
price = D['equally weighted']
t = np.array(t)
price = np.array(price)

def lwlr(tp,X,Y,k):
    #Locally weighted liner regression
    #k is a value to be taken as an input
    x = mat(X)
    y = mat(Y)
    m = shape(x)[0]
    weights = mat(eye(m))
    for j in range(m):
        dm = tp - x[j,:]
        weights(j,j) = exp(dm*dm.T/-2*k**2)
    res = tp*weights
    return res

def linearWeightedRegression(testA,xA,yA,k):
    m = shape(testA)[0]
    yH = zeros[m]
    for i in range(m):
        yH[i] = lwlr(testA[i],xA,yA,k)
    return yH

#Standard Regression
def StandRegress(xArr,yArr):
    x = mat(xArr)
    y = mat(yArr)
    xTx = xMat.T*xMat
    #Check for the singularity of xTx
    if np.linalg.inv(xTx) does not exist:
        lam
        


    
