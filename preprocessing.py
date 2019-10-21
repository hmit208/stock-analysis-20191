import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import nn, optim
from torch.autograd.variable import Variable
from sklearn.preprocessing import StandardScaler
from math import sqrt
from configs import *

df = pd.read_csv(
    './raw/FPT.csv')

# setting index as date
df['Date'] = pd.to_datetime(df.Date, format='%d/%m/%Y')
df.index = df['Date']

"""
plt.figure(figsize=(26, 8))
plt.plot(df['Close'], label='Close Price history')
print(df['Close'])
plt.show()
"""

data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=cols)
for i in range(0, len(data)):
    for col in cols:
        new_data[col][i] = data[col][i]

# setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

dataset = new_data.values

sma_short_time = new_data['Close'].rolling(window=short_time).mean()
sma_long_time = new_data['Close'].rolling(window=long_time).mean()
exp_short_time = new_data['Close'].ewm(span=short_time, adjust=False).mean()
exp_long_time = new_data['Close'].ewm(span=long_time, adjust=False).mean()

"""
plt.plot(data['Date'], new_data['Close'], label='Close')
plt.plot(data['Date'], rolling_mean, label='12 Day SMA', color='orange')
plt.plot(data['Date'], rolling_mean2, label='100 Day SMA', color='magenta')
plt.plot(data['Date'], exp1, label='12 Day EMA', color='gray')
plt.plot(data['Date'], exp2, label='100 Day EMA', color='pink')
plt.legend(loc='upper left')
plt.show()
"""

# Adding features
new_data['sma_short_time'] = sma_short_time
# new_data['sma_long_time'] = sma_long_time
new_data['exp_short_time'] = exp_short_time
# new_data['exp_long_time'] = exp_long_time

_input_dim = new_data.shape[1]
lstm_params['input_dim'] = _input_dim
# print("lstm_params['input_dim']: ", lstm_params['input_dim'])

new_data = new_data.dropna()
dataset = new_data.values

"""
def normalize_data(dataset):
    print("dataset: ", dataset.shape)
    scaler = StandardScaler()
    dataset_normalized = scaler.fit_transform(dataset)
    # dataset_normalized = scaler.fit(dataset)
    # print("Mean: ", dataset_normalized.mean_)
    # print("va: ", np.sqrt(dataset_normalized.var_))
    print("----dataset after normalized---\n")
    for data in dataset_normalized:
        print(data)
    # standardization the dataset and print the first 5 rows
    print("invert-----")
    inversed = scaler.inverse_transform(dataset_normalized)
    for data in inversed:
        print(data)

for i in range(5, len(dataset)):
    a = dataset[i - 5:i, :]
    # print("a: ", a)
    normalize_data(a)
    break


"""

# print(dataset[int(0.95 * len(dataset)):, :])

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

scaler_labels = MinMaxScaler(feature_range=(0, 1))
# print("####: ", dataset[: , 0])
# print("$$$: ", dataset)
scaled_label_data = scaler_labels.fit_transform(dataset[: , 0].reshape(-1, 1))
# print("00000")
# print(scaler_labels.inverse_transform(scaled_label_data))



# x_train, y_train = [], []

X = []
y = []

for i in range(time_prev, len(scaled_data)):
    if i + num_future_days <= len(scaled_label_data):
        X.append(scaled_data[i - time_prev:i, :])
        y.append(scaled_data[i: i + num_future_days, 0])
X, y = np.array(X), np.array(y)


X = torch.from_numpy(X)
y = torch.from_numpy(y)
y = y.view(len(y), num_future_days)

print(scaler_labels.inverse_transform(y))

# SPLIT = int(0.95 * len(X))
SPLIT = len(X) - 7
X_train = X[0: SPLIT, :]
y_train = y[0: SPLIT, ]
X_valid = X[SPLIT:, :]
y_valid = y[SPLIT:, ]
print("NUMBER VALID: ", len(y_valid))
dataset_train = TensorDataset(X_train, y_train)
dataset_valid = TensorDataset(X_valid, y_valid)

train_loader = DataLoader(dataset=dataset_train, batch_size=lstm_params['batch_size'], shuffle=False)
test_loader = DataLoader(dataset=dataset_valid, batch_size=lstm_params['batch_size'], shuffle=False)
