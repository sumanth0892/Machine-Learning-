def lwlr(xtest,xArr,yArr,k=1.0):
    xMat = mat(xArr)
    ymat = mat(yArr).T
    s = xmat.shape[0]
    weights = mat(eye(m))
    for i in range(s):
        testMat = xtest - xMat[i,:]
        weights[i,i] = exp(det(testMat)/(2*k*k))
    xTx = xMat.T * weights * xMat
    ws = xTx.I * xMat.T * weights * yArr
    return xtest*ws

def lwlrtest(testArr,xArr,yArr,k=1.0):
    m = testArr.shape[0]
    yHat = zeros[m]
    for j in range(m):
        yHat[j] = lwlr(testArr[j],xArr,yArr,k)
    return yHat
