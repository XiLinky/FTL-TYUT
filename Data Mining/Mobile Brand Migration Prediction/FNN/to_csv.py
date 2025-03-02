from torch import nn
import torch
from read_data import train_feature, train_target, test_data, config
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Network(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.BatchNorm1d(hidden1_dim),
            # nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(hidden1_dim, output_dim),
            # nn.BatchNorm1d(hidden2_dim),
            # nn.LeakyReLU(),
            # nn.Linear(hidden2_dim, hidden3_dim),
            # nn.BatchNorm1d(hidden3_dim),
            # nn.Dropout(),
            # nn.LeakyReLU(),
            # nn.Linear(hidden3_dim, output_dim)
            # nn.Softmax()
        )

    def forward(self, x):
        out = self.layer(x)
        return out


model = Network(train_feature.shape[1], 64, 1024, 512, train_target.shape[1])
model.to(device)
model.load_state_dict(torch.load('model_state.pth'))
model.train()
# model.eval()

# 用于映射手机品牌
affine = dict()
keys = [i for i in range(11)]

for key in keys:
    affine[key] = config['res'][key][8:]

pred = model(test_data)
pred = torch.argmax(pred, -1)
pred = list(pred.detach().numpy())

# 将索引换成手机品牌
for index, value in enumerate(pred):
    for key in affine:
        if value == key:
            pred[index] = affine[key]

res = pd.DataFrame(pred)
res.columns = ['新5G终端品牌']
ori = pd.read_csv('./ori_get_test.csv')
res = pd.concat([ori['用户标识'], res], axis=1)
print(res)
res.to_csv('result.csv', index=False)