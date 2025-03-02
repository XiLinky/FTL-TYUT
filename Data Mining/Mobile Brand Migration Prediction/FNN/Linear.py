from torch import nn
import torch
from read_data import train_feature, train_target


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

# 使用Xavier初始化权重
for line in model.layer:
    if type(line) == nn.Linear:
        nn.init.kaiming_uniform_(line.weight)