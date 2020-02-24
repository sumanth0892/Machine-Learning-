import os
import numpy as np
import pandas as pd
import datetime as datetime
pd.set_option('display.width',1000)
pd.set_option('display.max_columns',1000)
from sklearn.preprocessing import LabelEncoder
def warn(*args,**kwargs):
    pass
import warnings
warnings.warn = warn
np.seterr(all = 'ignore')

def getYearWeekDay(D):
    date = D.strip().split()
    date.pop(1)
    date = date[0].split('-')
    return datetime.date(int(date[0]),int(date[1]),int(date[2])).isocalendar()


X = pd.read_csv('sfTrain.csv')
x = pd.read_csv('sfTest.csv')
X = X.rename({'PdDistrict':'District','Category':'label','Dates':'Date'},axis = 1)
x = x.rename({'PdDistrict':'District','Dates':'Date'},axis = 1)
X = X.drop(columns = ['Resolution','Address','Descript','DayOfWeek'],axis = 1)
x = x.drop(columns = ['Id','Address','DayOfWeek'],axis = 1)
dates = X.Date.apply(getYearWeekDay)
year = []; week = []; day = []
for date in dates:
    year.append(date[0])
    week.append(date[1])
    day.append(date[2])
X['Year'] = year; X['Week'] = week; X['Day'] = day
del dates,year,week,day
dates = x.Date.apply(getYearWeekDay)
year = []; week = []; day = []
for date in dates:
    year.append(date[0])
    week.append(date[1])
    day.append(date[2])
x['Year'] = year; x['Week'] = week;x['Day'] = day
del dates,year,week,day
X = X.drop(columns = ['Date'],axis = 1)
x = x.drop(columns = ['Date'],axis = 1)
leCat = LabelEncoder()
leCat.fit(X.label)
X.label = leCat.transform(X.label)
dists = list(set(X.District))
leDist = LabelEncoder()
leDist.fit(dists)
X.District = leDist.transform(X.District)
x.District = leDist.transform(x.District)
X = X.astype('object')
x = x.astype('object')
targets = list(X.label)
targets = [str(i) for i in targets]
X = X.drop(columns = ['label'],axis = 1)
testSet = x[:10000]
XTrain = np.array(X)
YTrain = np.array(targets)
xTest = np.array(x)




#See the transformed data
X.info()
print(X[:10])
x.info()
print(x[:10])



#K-nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors = 5)
neigh.fit(X,targets)
preds = neigh.predict(x)
print(leCat.inverse_transform(preds[:10]))

"""
from keras import models,layers
from keras.utils.np_utils import to_categorical
YTrain = to_categorical(YTrain)
model = models.Sequential()
model.add(layers.Dense(32,activation='relu',input_shape=(5,)))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(39,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',metrics = ['acc'],
              optimizer = 'rmsprop')
H = model.fit(XTrain,YTrain,epochs = 25,batch_size = 256,verbose = 2)

"""
