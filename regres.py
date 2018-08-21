from numpy import *

def loadDataSet(fileName):
    numfeat = len(open(fileName).readline().split('\t')) -1
    dataMat = []; labelMat=[]
    fr = open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine = line.strip().split('\t')
        for i in range(numfeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat.labelMat

def regression (xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    #if (xTx.I== 0):
     #   print("Inverse doesn't exist")
      #  return
    w = xTx.I * (xMat.T * yMat)
    return w

def error(xArr,yArr,w):
    xMat = mat(xArr)
    yMat = mat(yArr)
    ypred = xArr.T * w
    error = 0
    a=ypred - yMat
    
    error = sum(a**2) 
    return error,ypred

