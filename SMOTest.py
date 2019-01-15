def CalcErr(x,set,a,y,b):
    Ex = a*y*x*dataSet + b
    return float(Ex)

def selectRand(i,m):
    j = random.rand(0,m)
    while (i==j):
        j = random.rand(0,m)
    return j

def clipAlpha(
    
def smoSimple(dataSet,C,toler,max_iters=500):
    X = dataSet[:,0:n-1]; Y = dataSet[:,-1]
    #X is of dimension (m,n-1). Y is of size (m,1)
    alphas = mat(zeros((m,1))); b=0
    num_iters=0
    while (num_iters<max_iters):
        num_changed_alphas=0
        for i in range(m):
            Ei = CalcErr(X[i]) - float(Y[i]) #Error function
            if ((Y[i]*Ei<-toler && alphas[i]<C)||(Y[i]*Ei>tol && alphas[i]>0)):
                j = selectRand(i,m) #Random selction function
                Ej = CalcErr(X[j],X,alphas[i],Y[i]) - float(Y[j]) #Call error function again
                alphaIold = alphas[i].copy();
                alphaJold = alphas[j].copy();
                L,H = calcThresh(alphas[i],alphas[j],C)
                if L==H: continue
                eta = ()
                alphas[j] = () #Support function to clip new value for alpha[j]
                if (abs(alphas[j] - alphaJold))<0.00001:
                    continue
                alphas[i] = () #Support function for alpha
                b1=
                b2=
                b=
                num_changed_alphas+=1
        if (num_changed_alphas==0):
            num_iters+=1
        else:
            num_iters=0
    
