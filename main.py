#data exploration
import pandas as pd

df = pd.read_csv('Clean_Dataset.csv')
df.airline.value_counts()
df.source_city.value_counts()
df.destination_city.value_counts()
df.departure_time.value_counts()
df.arrival_time.value_counts()
df.stops.value_counts()
df['class'].value_counts()
df['duration'].min()
df['duration'].max()
df['duration'].median()


#preprocessing
df = df.drop('Unnamed: 0', axis=1)
df = df.drop('flight', axis = 1)

df['class'] = df['class'].apply(lambda x:1 if x == 'Business' else 0)
df.stops = pd.factorize(df.stops)[0]
df = df.join(pd.get_dummies(df.airline, prefix='airline')).drop('airline', axis=1)
df = df.join(pd.get_dummies(df.source_city, prefix='source_city')).drop('source_city', axis=1)
df = df.join(pd.get_dummies(df.destination_city, prefix='destination_city')).drop('destination_city', axis=1)
df = df.join(pd.get_dummies(df.arrival_time, prefix='arrival')).drop('arrival_time', axis=1)
df = df.join(pd.get_dummies(df.departure_time, prefix='departure')).drop('departure_time', axis=1)

#training 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

x, y = df.drop('price', axis = 1), df.price
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

reg = RandomForestRegressor()
reg.fit(x_train, y_train)

reg.score(x_test, y_test)

import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = reg.predict(x_test)

print('R2', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))
print('MSE', mean_squared_error(y_test, y_pred))
print('RMSE', math.sqrt(mean_squared_error(y_test, y_pred)))


#Evaluation
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Flight Price")
plt.ylabel("Predicted Flight Price")
plt.title('Prediction vs Actual')

importances = dict(zip(reg.feature_names_in_, reg.feature_importances_))
sorted_importances = sorted(importances.items(), key = lambda x: x[1], reverse = True)
sorted_importances

plt.figure(figsize=(10,6))
plt.bar([x[0] for x in sorted_importances[:5]], [x[1] for x in sorted_importances[:5]])

#fine tuning

from sklearn.model_selection import GridSearchCV
''''
param_grid = {
    'n_estimators' : [100, 200, 300],
    'max_depth': [None, 10,20, 30],
    'min_samples_split' : [2, 5,10],
    'min_samples_leaf' : [1,2,4],
    'max_features' : ['auto', 'sqrt']
}

grid_search = GridSearchCV(reg, param_grid, cv = 5)
grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_
'''

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators' : randint(100, 300),
    'max_depth': [None, 10,20, 30, 40, 50],
    'min_samples_split' : randint(2,11),
    'min_samples_leaf' : randint(1,5),
    'max_features' : [1.0, 'auto', 'sqrt']
}

reg = RandomForestRegressor(n_estimators=-1)
random_search = RandomizedSearchCV(estimator= reg, param_distributions= param_dist, n_iter = 2, cv = 3,
scoring = 'neg_mean_squared_error', verbose = 2, random_state =10, n_jobs =-1)

random_search.fit(x_train, y_train)

best_regressor = random_search.best_estimator_
best_regressor.score(x_test, y_test)

y_pred = best_regressor.predict(x_test)

best_regressor.predict()