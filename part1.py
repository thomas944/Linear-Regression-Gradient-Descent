import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# Data processing
FILE_PATH = 'https://raw.githubusercontent.com/thomas944/ml-assignment-1/main/student-mat.csv'

df = pd.read_csv(FILE_PATH, sep=';', skiprows=0, header=None)
df.set_axis(df.iloc[0], axis=1, inplace=True)
df = df[1:]
df.rename(columns=
  {
  'Fedu': 'father_edu', 'Medu': 'mother_edu', 
  'G3': 'final_grade', 'romantic': 'relationship'
  }, 
  inplace=True
)

features = [
  'father_edu', 'mother_edu', 'traveltime', 
  'studytime', 'failures', 'schoolsup', 
  'famsup', 'paid', 'higher', 'internet', 
  'relationship', 'goout', 'Dalc', 'Walc', 
  'absences', 'final_grade'
]

data = df[features]

categorical_cols = data.select_dtypes(include='object').columns

# Convert Categorical Variables to Numbers
for col in categorical_cols:
    data[col] = data[col].replace({'yes': 1, 'no': 0})

data[list(features)] = data[list(features)].astype(float)

X = data.drop('final_grade', axis=1)
Y = data['final_grade']


# Randomly choose weights and bias value
def initialize(num_features):
  bias = random.random() 
  weights = np.random.randn(num_features) 
  return bias, weights

# Predict new y value
def predict_y(bias, weights, X):
  X_values = np.array(X)
  return bias + np.dot(X_values, weights)

# Calculate MSE
def get_cost(Y, Y_hat):
  Y_res = Y - Y_hat
  return np.mean(Y_res ** 2)

# Update Weights
def update(X, Y, Y_hat, old_bias, old_weights, learning_rate):
  db = (np.mean(Y_hat - Y) * 2)
  dw = (np.dot((Y_hat - Y), X) * 2) / len(Y)
  new_bias = old_bias - learning_rate * db
  new_weights = old_weights - learning_rate * dw
  return new_bias, new_weights

#Calculate R Squared Value
def calculate_r_squared(Y_true, Y_pred):
  mean_true = np.mean(Y_true)
  tss = np.sum((Y_true - mean_true) ** 2)
  rss = np.sum((Y_true - Y_pred) ** 2)
  r_squared = 1 - (rss / tss)
  return r_squared

# Gradient descent implementation
def gradient_descent(X_train, X_test, Y_train, Y_test, learning_rate, num_iterations):
  bias, weights = initialize(X_train.shape[1])
  mse_history_train = pd.DataFrame(columns=['iteration','cost','r_squared'])
  mse_history_test = pd.DataFrame(columns=['iteration','cost','r_squared'])
  array_index = 0
  for iteration in range(num_iterations):
    Y_hat_train = predict_y(bias, weights, X_train)
    cost_train = get_cost(Y_train, Y_hat_train)
    r_squared_train = calculate_r_squared(Y_train,Y_hat_train)
    bias, weights = update(X_train, Y_train, Y_hat_train, bias, weights, learning_rate)

    Y_hat_test = predict_y(bias, weights, X_test)
    cost_test = get_cost(Y_test, Y_hat_test)
    r_squared_test = calculate_r_squared(Y_test,Y_hat_test)
    if iteration % 10 == 0:
      mse_history_train.loc[array_index]=[iteration,cost_train,r_squared_train]
      mse_history_test.loc[array_index]=[iteration,cost_test,r_squared_test]
      array_index += 1
  return mse_history_train, mse_history_test, bias, weights



# 80/20 Split for train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Parameters 
learning_rate = 0.001
num_iterations = 10000

mse_history_test, mse_history_train, bias, weights = gradient_descent(X_train, X_test, Y_train, Y_test, learning_rate, num_iterations)
Y_train_pred = predict_y(bias, weights, X_train)

mse_train = get_cost(Y_train, Y_train_pred)

r_squared = calculate_r_squared(Y_train, Y_train_pred)

print("MSE for Train Data:", mse_history_train['cost'].iloc[-1])
print("MSE for Test Data:", mse_history_test['cost'].iloc[-1])
print("R-Squared Value:", r_squared)
print("Weights of Model:", weights)

## MSE Train VS Iteration Graph ##
# plt.plot(mse_history_train['iteration'], mse_history_train['cost'], label='alpha=0.001')
# plt.ylabel('MSE Train')
# plt.xlabel('Number of iterations')
# plt.title('MSE Train Vs. Iterations')
# plt.show()

## MSE Test VS Iteration Graph ##
# plt.plot(mse_history_test['iteration'], mse_history_test['cost'], label='alpha=0.001')
# plt.ylabel('MSE Test')
# plt.xlabel('Number of iterations')
# plt.title('MSE Test Vs. Iterations')
# plt.show()


# mse_history_003, bias_003, weights_003 = gradient_descent(X_train, Y_train, learning_rate=0.003, num_iterations=10000)
# Y_train_pred_003 = predict_y(bias_003, weights_003, X_train)
# mse_train_003 = get_cost(Y_train, Y_train_pred_003)
# r_squared_003 = calculate_r_squared(Y_train, Y_train_pred_003)

# mse_history_005, bias_005, weights_005 = gradient_descent(X_train, Y_train, learning_rate=0.005, num_iterations=10000)
# Y_train_pred_005 = predict_y(bias_005, weights_005, X_train)
# mse_train_005 = get_cost(Y_train, Y_train_pred_005)
# r_squared_005 = calculate_r_squared(Y_train, Y_train_pred_005)

## MSE Train VS Iterations for Different Learning Rate ##
# plt.plot(mse_history['iteration'], mse_history['cost'], label='alpha=0.001')
# plt.plot(mse_history_005['iteration'], mse_history_005['cost'], label='alpha=0.005')
# plt.plot(mse_history_003['iteration'], mse_history_003['cost'], label='alpha=0.003')
# plt.legend()
# plt.ylabel('MSE')
# plt.xlabel('Number of iterations')
# plt.title('MSE Train Vs. Iterations for Different Learning Rate')


## R Squared Train VS Iterations for Different Alpha Values ##
# plt.plot(mse_history['iteration'], mse_history['r_squared'], label='alpha=0.001')
# plt.plot(mse_history_005['iteration'], mse_history_005['r_squared'], label='alpha=0.005')
# plt.plot(mse_history_003['iteration'], mse_history_003['r_squared'], label='alpha=0.003')
# plt.legend()
# plt.ylabel('R_Squared')
# plt.xlabel('Number of iterations')
# plt.title('R Squared Vs. Iterations for Different Alpha Values')