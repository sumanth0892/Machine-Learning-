def distEuc(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

def createCent(dataSet,k):
    n = shape(dataSet)[0]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minj = min(dataSet[:,j])
        maxj = max(dataSet[:,j])
        rangej = maxj - minj
        centroids[:,j] = minj+rangej*rand.random(k,1)
    return centroids

def kMeans(dataSet,k,centFunc=createCent,distFunc=distEuc):
    m,n = shape(dataSet); cluster = mat(zeros((m,2)))
    centroid = centFunc(dataSet,k)
    while clusterChanged:
        clusterChanged=False
        for i in range(m):
            minDist=inf; minIndex=-1
            for j in range(k):
                distJI = distFunc(dataSet[i,:],centroid[j,:])
                if distJI<minDist:
                    minDist=distJI
                    minIndex=j
            if cluster[i,0]!=minIndex: clusterChanged=True
            cluster[i,:] = minIndex,minDist**2
        print(centroid)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(cluster[:,0].A==cent)[0]]
            centroid[cent,:] = mean(ptsInClust,axis=0)
    return centroid,cluster

import matplotlib.pyplot as plt
def plotcluster(dataSet,k):
    centroids,clusters = kMeans(dataSet,k)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    ax0 = fig.add_axes(rect,label='ax0',**axprops)
    ax1 = fig.add_axes(rect,label='ax1',frameon=False)
    dMat = mat(dataSet)
    for i in range(cluster):
        ptsInClust = dMat[nonzero(cluster[:,0].A==i)[0],:]
        mStyle = markers[i%len(markers)]
        ax1.scatter(ptsInClust[:,0].flatten().A[0],\
                    ptsInClust[:,1].flatten().A[0],marker=mStyle,s=90)
    ax1.scatter(centroid,marker='+',s=300)
    plt.show()
    
        
