import torch
from torch import nn, optim
from Linear import model
from read_data import train_feature, train_target, validation_feature, validation_target


batch_size = 64
num_epochs = 100
lr = 3e-6
MSE_list = []
# t_acc = 0
# v_acc = 0

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

try:
    model.load_state_dict(torch.load('./model_state.pth'))
except FileNotFoundError:
    print("File is not found.")

for epoch in range(num_epochs):
    t_acc = 0
    v_acc = 0
    model.train()
    for i in range(0, train_feature.shape[0], batch_size):
        end = i + batch_size
        if end > (train_feature.shape[0] - 1):
            end = train_feature.shape[0] - 1

        mini_batch = train_feature[i:end]
        mini_batch_target = train_target[i:end]

        # forward
        out = model(mini_batch)
        loss = criterion(out, mini_batch_target)

        # reset grad
        optimizer.zero_grad()

        # backward
        loss.backward()
        optimizer.step()

    # for parameters in model.parameters():
    #     print(parameters)
    #     print(parameters.grad.data)

    # train_data计算loss和acc
    pred_t = model(train_feature)
    pred_t = pred_t.squeeze()
    # 因为tensor匹配会出现0 == 0的情况，只能用argmax做
    pred_t = torch.argmax(pred_t, -1)
    # criterion需要两个float
    pred_t = pred_t.float()
    train_target = torch.argmax(train_target, -1)
    train_target = train_target.float()
    t_loss = criterion(pred_t, train_target)
    # validation_data计算loss和acc  一切同上
    pred_v = model(validation_feature)
    pred_v = pred_v.squeeze()
    pred_v = torch.argmax(pred_v, -1)
    pred_v = pred_v.float()
    validation_target = torch.argmax(validation_target, -1)
    validation_target = validation_target.float()
    v_loss = criterion(pred_v, validation_target)
    acc1 = (pred_t == train_target).sum()
    acc2 = (pred_v == validation_target).sum()
    t_acc += acc1 / train_feature.shape[0]
    v_acc += acc2 / validation_feature.shape[0]
    # float转long 为了criterion能计算
    train_target = train_target.to(dtype=torch.long)
    validation_target = validation_target.to(dtype=torch.long)
    # 将argmax变回one-hot+float
    train_target = torch.nn.functional.one_hot(train_target, num_classes=11)
    train_target = train_target.to(dtype=torch.float32)
    validation_target = torch.nn.functional.one_hot(validation_target, num_classes=11)
    validation_target = validation_target.to(dtype=torch.float32)
    print(
        f"epoch:{epoch + 1}/{num_epochs} train_loss: {t_loss.data} train_acc: {t_acc} validation_loss: {v_loss.data} validation_acc: {v_acc}")
    # save
    torch.save(model.state_dict(), './model_state.pth')
