from torch import nn
import torch
from read_data2 import train_features

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Network(nn.Module):
    def __init__(self, input_dim, hidden_1, hidden_2, output_dim, weight_decay=0.01):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim)
            # nn.BatchNorm1d(hidden_1),
            # nn.ReLU(),
            # nn.Linear(hidden_1, output_dim)
        )

        # self.weight_decay = weight_decay

    def forward(self, x):
        out = self.layers(x)
        return out


model = Network(train_features.shape[1], 256, 32, 34)  # output_dim是17+17
model.to(device)

# 使用Xavier初始化权重
for line in model.layers:
    if type(line) == nn.Linear:
        nn.init.kaiming_uniform_(line.weight)
        # nn.init.xavier_uniform_(line.weight)
