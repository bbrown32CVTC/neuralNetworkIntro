# This is a Python script for the introduction of Neural Networks using PyTorch.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('CSC 419 HW 4 Pima Indians Diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]

# Convert data to 32-bit PyTorch tensors and shape to n x 1 matrix to avoid processing issues
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Define the fully-connected Sequential Model with three layers using the PyTorch Linear class, ReLU activation function, and Sigmoid function
model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

# model = nn.Sequential(
#     nn.Linear(8, 100),
#     nn.ReLU(),
#     nn.Linear(100, 100),
#     nn.ReLU(),
#     nn.Linear(100, 100),
#     nn.ReLU(),
#     nn.Linear(100, 1),
#     nn.Sigmoid()
# )
# print(model)

# # Verbose way of defining the model
# class PimaClassifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden1 = nn.Linear(8, 12)
#         self.act1 = nn.ReLU()
#         self.hidden2 = nn.Linear(12, 8)
#         self.act2 = nn.ReLU()
#         self.output = nn.Linear(8, 1)
#         self.act_output = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.act1(self.hidden1(x))
#         x = self.act2(self.hidden2(x))
#         x = self.act_output(self.output(x))
#         return x
#
# model = PimaClassifier()
# print(model)

# Training Preparation - Loss Metric and Optimizer
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training a Model with Training Loops - Epoch and Batch
n_epochs = 100
# n_epochs = 200
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

# Evaluate the Model
# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)

accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")
