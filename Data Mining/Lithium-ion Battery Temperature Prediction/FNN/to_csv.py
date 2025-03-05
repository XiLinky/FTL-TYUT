from read_data2 import test
import pandas as pd
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Network(nn.Module):
    def __init__(self, input_dim, hidden_1, hidden_2, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # nn.BatchNorm1d(hidden_1),
            # nn.ReLU(),
            # nn.Linear(hidden_1, output_dim)
        )

    def forward(self, x):
        out = self.layers(x)
        return out


model = Network(51, 256, 32, 34)  # output_dim是17+17
model.to(device)
model.load_state_dict(torch.load('model.pth'))
model.train()
# model.eval()

pred = model(test)
res = pd.DataFrame(pred.detach().numpy())
res.columns = ['上部温度'+str(i+1) for i in range(17)] + ['下部温度'+str(i+1) for i in range(17)]
ori = pd.read_csv(
    '../锂离子电池生产参数调控及生产温度预测挑战赛公开数据-初赛/锂离子电池生产参数调控及生产温度预测挑战赛公开数据-初赛/test.csv')
res = pd.concat([ori['序号'], res], axis=1)
print(res)
res.to_csv('result_wtf5_256batch_500.csv', index=False)