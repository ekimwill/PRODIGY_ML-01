import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

data_train = pd.read_csv(
    r'C:\Users\Mailaf Tewodros\Desktop\Soft\AI\ML\project mike\project2_home_predict\dataset\train.csv')
data_test = pd.read_csv(
    r'C:\Users\Mailaf Tewodros\Desktop\Soft\AI\ML\project mike\project2_home_predict\dataset\test.csv')

X_train = data_train[['LotArea', 'BedroomAbvGr', 'FullBath']].values.astype(np.float32)
y_train = data_train['SalePrice'].values.astype(np.float32).reshape(-1, 1)

# X_test = data_test[['LotArea', 'BedroomAbvGr', 'FullBath']].values.astype(np.float32)
# y_test = data_test['SalePrice'].values.astype(np.float32).reshape(-1, 1)


X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

X_train_tensor = torch.tensor(X)
Y_train_tensor = torch.tensor(y)

dataset = TensorDataset(X_train_tensor, Y_train_tensor)
batch_size = 20
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


class SVRModel(nn.Module):
    def __init__(self, input_size):
        super(SVRModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # Output size is always 1 for regression

    def forward(self, x):
        return self.linear(x)


# input_size = 3  # Number of features
output_size = 1  # Single output (price)
# model = LinearRegressionModel(input_size)
input_size = 3  # Number of features
model = SVRModel(input_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=100)


num_epochs = 1000000
train_losses = []
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor.unsqueeze(1))  # Unsqueeze y_train_tensor to match output shape
    loss_ = criterion(outputs, Y_train_tensor)
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #     if (epoch+1) % 10000 == 0:
    #         optimizer = torch.optim.RMSprop(model.parameters(), lr=10)

    #     train_losses.append(loss.item())
    train_losses.append(loss_.item())
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')




