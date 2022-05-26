import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics


dataset= pd.read_csv('train.csv', index_col="Id")
print(dataset.head())
percent = int( ((100-5)/100) * dataset.shape[0] )
dataset = dataset.dropna(axis=1, thresh=dataset.shape[0]*0.85, how='all')
print(dataset.head())

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
X = pd.get_dummies(dataset, drop_first=True)
X.fillna(X.mean(), inplace=True)
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train, y_train)
print(regressor.score(X_train, y_train))