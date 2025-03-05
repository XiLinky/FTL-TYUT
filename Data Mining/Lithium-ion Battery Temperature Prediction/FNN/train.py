from Linear import model
from read_data2 import train_features, train_labels, validation_labels, validation_features
from torch import nn, optim
import torch

batch_size = 256
MAE_list = []
num_epochs = 500


# 损失函数与优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
target_loss = nn.L1Loss()


def forward_L1(model, feature, target):
    model.eval()
    pred = model(feature)
    pred = pred.squeeze()
    res = target_loss(pred, target).item()
    return res


def forward_L2(model, feature):
    pred = model(feature)
    pred = pred.squeeze()
    return pred


try:
    model.load_state_dict(torch.load('model.pth'))
except FileNotFoundError:
    print("File is not found.")

for epoch in range(num_epochs):
    model.train()
    for i in range(0, train_features.shape[0], batch_size):
        end = i + batch_size
        if i + batch_size > train_features.shape[0] - 1:
            end = train_features.shape[0] - 1
        mini_batch = train_features[i:end]
        mini_batch_target = train_labels[i:end]
        # forward
        pred = forward_L2(model, mini_batch)
        loss = criterion(pred, mini_batch_target)
        if torch.isnan(loss):
            break
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_mae = forward_L1(model, train_features, train_labels)
    Validation_mae = forward_L1(model, validation_features, validation_labels)
    print(f"epoch:{epoch + 1}/{num_epochs} Train_MAE: {train_mae} Validation_MAE: {Validation_mae}")
    MAE_list.append((train_mae, Validation_mae))
    torch.save(model.state_dict(), 'model.pth')

