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

print("=== DATA EXPLORATION COMPLETE ===")
print("Explored airline distribution, city patterns, time distributions, and duration statistics")
print("Starting preprocessing")


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

print("=== DATA PREPROCESSING COMPLETE ===")
print("Removed unnecessary columns, encoded categorical variables, and created dummy variables for ML")
print("Starting to train model")

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

print("=== INITIAL MODEL TRAINING COMPLETE ===")
print("Trained baseline Random Forest model and evaluated performance metrics")
print("Starting Model Evaluation")


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

print("=== MODEL EVALUATION COMPLETE ===")
print("Created scatter plot of predictions vs actual values and feature importance visualization")
print("Starting Fine Tuning")

#fine tuning

from sklearn.model_selection import GridSearchCV
'''
param_grid = {
    'n_estimators' : [100, 200, 300],
    'max_depth': [None, 10,20, 30],
    'min_samples_split' : [2, 5,10],
    'min_samples_leaf' : [1,2,4],
    'max_features' : ['sqrt', 'log2']  # Fixed: removed 'auto' as it's not valid
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
    'max_features' : [1.0, 'sqrt', 'log2']  # Fixed: removed 'auto' as it's not valid
}

reg = RandomForestRegressor(n_estimators=100)  # Fixed: n_estimators cannot be -1
random_search = RandomizedSearchCV(estimator= reg, param_distributions= param_dist, n_iter = 2, cv = 3,
scoring = 'neg_mean_squared_error', verbose = 2, random_state =10, n_jobs =-1)

random_search.fit(x_train, y_train)

best_regressor = random_search.best_estimator_
best_regressor.score(x_test, y_test)

y_pred = best_regressor.predict(x_test)

# Fixed: predict() requires X argument
print("Best regressor predictions:")
print(best_regressor.predict(x_test)[:10])  # Show first 10 predictions

# Show the best parameters found
print("\nBest parameters found:")
print(random_search.best_params_)

# Show the best score
print(f"\nBest cross-validation score: {random_search.best_score_:.4f}")

# Make predictions on test set and evaluate
y_pred_best = best_regressor.predict(x_test)
print(f"\nBest model performance on test set:")
print(f'R2: {r2_score(y_test, y_pred_best):.4f}')
print(f'MAE: {mean_absolute_error(y_test, y_pred_best):.4f}')
print(f'MSE: {mean_squared_error(y_test, y_pred_best):.4f}')
print(f'RMSE: {math.sqrt(mean_squared_error(y_test, y_pred_best)):.4f}')

# Save the best model for web deployment
import pickle
with open('flight_model.pkl', 'wb') as f:
    pickle.dump(best_regressor, f)
print("\nModel saved as 'flight_model.pkl' for web deployment")

print("=== HYPERPARAMETER TUNING COMPLETE ===")
print("Completed RandomizedSearchCV optimization and evaluated best model performance")
print("=== ALL PROCESSES COMPLETED SUCCESSFULLY ===")

