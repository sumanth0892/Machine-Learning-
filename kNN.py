from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    ClassCount= {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        ClassCount[voteIlabel] = ClassCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(ClassCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))
    classLabelVector=[]
    fr = open(filename)
    index=0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        index+=1
    return returnMat,classLabelVector

def normalizedData(dataSet):
    minValues = dataSet.min(0)
    maxValues = dataSet.max(0)
    ranges = maxValues - minValues
    normalizedSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normalizedSet = dataSet - tile(minValues,(m,1))
    normalizedSet = normalizedSet/tile(ranges,(m,1))
    return normalizedSet,minValues,ranges

def datingClassTest():
    hoRatio = 0.10
    datingMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = normalizedData(datingMat)
    m = normmat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\datingLabels[numTestVecs:m],3)
        print ("The classifier came back with: %d, the real answer is: %d"\ %(classifierResult,datingLabels[i]))
        if (classifierResult!= datingLabels[i]):
            errorCount+=1.0
    print ("The total error rate is: %f" %(errorCount/float(numTestVecs)))
    
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input(\"percentage of time spent playing video games?"))
    ffMiles = float(input(\"FF Miles earned?"))
    iceCream = float(input(\"Liters of ice cream consumer per year?"))
    datingMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = mormalizedData(datingMat)
    inArr = array([ffmiles,percentTats,iceCream])
    classifierResult = classify0((inArr-\minVals)/ranges,normMat,datingLabels,3)
    print ("You will probably like this person: ",\resultList[classifierResult-1])
    
                        
