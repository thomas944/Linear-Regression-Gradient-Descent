import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns


# Data processing
FILE_PATH = 'https://raw.githubusercontent.com/thomas944/ml-assignment-1/main/student-mat.csv'

df = pd.read_csv(FILE_PATH, sep=';', skiprows=0, header=None)
df.set_axis(df.iloc[0], axis=1, inplace=True)
df = df[1:]
df.rename(columns={'Fedu':'father_edu','Medu':'mother_edu','G3':'final_grade','romantic':'relationship'},inplace=True)

features = ['father_edu', 'mother_edu', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'higher', 'internet', 'relationship', 'goout', 'Dalc', 'Walc', 'absences', 'final_grade']
data = df[features]

categorical_cols = data.select_dtypes(include='object').columns

# Convert Categorical Variables to Numbers
for col in categorical_cols:
  data[col] = data[col].replace({'yes': 1, 'no': 0})

data[list(features)] = data[list(features)].astype(float)

X = data.drop('final_grade', axis=1)
Y = data['final_grade']

# 80/20 Split for train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Parameters
learning_rate = 0.001
num_iterations = 10000

# Gradient Descent Implementation
def gradient_descent(X_train, X_test, Y_train, Y_test, num_iterations, learning_rate):

  model = linear_model.SGDRegressor(max_iter=num_iterations, random_state=42, alpha=learning_rate)
  mse_history_train = []
  mse_history_test = []

  for iteration in range(num_iterations):
    model.partial_fit(X_train, Y_train)
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    cost_train = mean_squared_error(Y_train, Y_train_pred)
    cost_test = mean_squared_error(Y_test, Y_test_pred)
    mse_history_train.append(cost_train)
    mse_history_test.append(cost_test)

  score = model.score(X_train,Y_train)
  weights = model.coef_
  return mse_history_train, mse_history_test, score, weights

mse_history_train, mse_history_test, score, weights = gradient_descent(X_train, X_test, Y_train, Y_test, num_iterations, learning_rate)

print("MSE for Train Data:", mse_history_train[-1])
print("MSE for Test Data:", mse_history_test[-1])
print("R-Squared:", score)
print("Weights of Model ", weights)


## Scatter plot implementation ##
# plt.scatter(Y_test, Y_test_pred)
# plt.xlabel("Actual Y")
# plt.ylabel("Predicted Y")
# plt.title('Y-Test VS Y-Predicted')
# plt.grid()
# plt.show()

## Correlation matrix implementation ##
# corr = data.corr()
# plt.figure(figsize=(11,8))
# sns.heatmap(corr,cmap=sns.diverging_palette(220, 10),annot=True,fmt='.2f',)
# plt.title('Correlation Matrix')
# plt.show()

# mse_history_train_003, mse_history_test_003, score, weights = gradient_descent(X_train, X_test, Y_train, Y_test, num_iterations, .003)
# mse_history_train_005, mse_history_test_005, score, weights = gradient_descent(X_train, X_test, Y_train, Y_test, num_iterations, .005)

## MSE Train Vs. Iterations for Different Learning Rate ##
# history = []
# for i in range(0,10000):
#   history.append(i)
# plt.plot(history, mse_history_train, label='alpha=0.001')
# plt.plot(history, mse_history_train_003, label='alpha=0.003')
# plt.plot(history, mse_history_train_005, label='alpha=0.005')
# plt.legend()
# plt.ylabel('MSE')
# plt.xlabel('Number of iterations')
# plt.title('MSE Train Vs. Iterations for Different Learning Rate')

## MSE Test Vs. Iterations for Different Learning Rate ##
# plt.plot(history, mse_history_test, label='alpha=0.001')
# plt.plot(history, mse_history_test_003, label='alpha=0.003')
# plt.plot(history, mse_history_test_005, label='alpha=0.005')
# plt.legend()
# plt.ylabel('MSE')
# plt.xlabel('Number of iterations')
# plt.title('MSE Test Vs. Iterations for Different Learning Rate')