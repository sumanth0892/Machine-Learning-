#A program to compute weights in a Linear regression model
from numpy import *
from pandas import *
import matplotlib.pyplot as plt
class CleanData(object):
    def __init__(column,length):
        self.data = column.dropna()
        self.length = length

    def clean(self):
        x = self.data
        y=[]
        for i in range(self.length):
            a = x[i].replace(',','.')
            y.append(float(a))

        return array([y])

def linReg(X,Y):
    X2 = X.T
    w = linalg.inv(X2*X) * X2 * Y #Or Y.T
    return w

def error(w,Y,X):
    y = len(Y)
    e=[]
    for i in len(y):
        x = Y(i) - X[i,i] * w
        e.append(x**2)

    return sum(e)

def predictions(pred,w):
    YPred = pred*w
    return YPred

InputData = read_table(' ',',')
v1=
v2=
l1=
l2
xMat = mat(v1,v2,...)
yMat = mat(Y)
yMat = yMat.T
weights = linReg(xMat,yMat)
Error = error(weights,xMAt,yMat)
pred_data = input("Enter the data to be predicted")
plt.scatter(yMat,predictions(pred_data,weights),'bo','r--')
plt.grid(True,color='black')
plt.show()


