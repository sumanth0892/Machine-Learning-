#import os
#import tarfile
#from six.moves import urllib
#download_root = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
#HOUSING_PATH = os.path.join("datasets","housing")
#HOUSING_URL = download_root + "datasets/housing/housing.tgz"

#def fetch_housing_data(housing_url = HOUSING_URL,housing_path=HOUSING_PATH):
#    if not os.path.isdir(housing_path):
#        os.makrdirs(housing_path)
#
#    tgz_path = os.path.join(housing_path,"housing.tgz")
#    urllib.request.urlretrieve(housing_url,tgz_path)
#    housing_tgz = tarfile.open(tgz_path)
#    housing_tgz.extractall(path=housing_path)
#    housing_tgz.close()
import numpy as np
import pandas as pd

def load_housing_data():
    return pd.read_csv('housing.csv')
housing = load_housing_data()

import numpy as np
def split_train_test(data,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

#The above procedure can also be done using a simple inbuilt function in Scikit-learn
def split_data(data,size = 0.2,random_state=45):
    from sklearn.model_selection import train_test_split
    train_set,test_set = train_test_split(data,test_size=size,random_state=random_state)
    return train_set,test_set

#stratified shuffling
from sklearn.model_selection import StratifiedShuffleSplit
def Stratified_Split(housing,test_size = 0.2,random_size=42):
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    housing['income_cat'] = np.ceil(housing['median_income']/1.5)
    housing['income_cat'].where(housing['income_cat']<5,5.0,inplace=True)
    for train_index,test_index in split.split(housing,housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    working_train_set = strat_train_set.copy    
    return strat_train_set,strat_test_set,working_train_set

housing.dropna(subset=['total_bedrooms']) #Missing values
housing.drop('total_bedrooms',axis=1)
median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median,inplace=True)

#This can also be done with Imputer in Scikit-learn
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')
housing_num = housing.drop('ocean_proximity',axis=1)
imputer.fit(housing_num)
print(housing_num.median().values)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X,columns = housing_num.columns)


#Transformation Pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline([
    ('imputer',Imputer(strategy='median')),
    ('attribs_adder',CombinedAttributesAdder()),
    ('std_scaler',StandardScaler())])
housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline',num_pipeline),
    ('cat_pipeline',cat_pipeline)])

housing_prepared = full_pipeline.fit_transform(housing)

print(housing_prepared.shape)


#Training and evaluation on the training set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
housin_preds1 = lin_reg.predict(housing_prepared)
mse1 = mean_squared error(housing_labels,housing_preds1)
print("Mean Square error using Linear regression is: %f" %sqrt(mse1))

#Tree regression
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)
housin_preds2 = tree_reg.predict(housing_prepared)
mse2 = mean_squared error(housing_labels,housing_preds1)
print("Mean Square error using Tree regression is: %f" %sqrt(mse2))








