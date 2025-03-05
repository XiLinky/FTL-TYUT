import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class Config:
    timestep = 30  # 时间步长
    batch_size = 16
    hidden_dim = 64
    output_dim = 1
    num_layers = 2
    epochs = 50
    lr = 0.001
    dropout = 0.3
    embedding_dim = 10
    save_path = 'cnn_lstm_model.pth'
    data_path = './used_car_train_20200313.csv'


config = Config()


# 数据预处理
def preprocess_data():
    df = pd.read_csv(config.data_path, sep=' ', nrows=1000)
    df.drop(['notRepairedDamage', 'brand'], axis=1, inplace=True)
    df = df.dropna()

    # 类别特征处理
    cate_cols = ['fuelType', 'gearbox', 'offerType']
    for col in cate_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # 连续特征标准化
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, cate_cols, num_cols, scaler


# 构建时间窗口数据
def create_sequences(data, timestep):
    sequences = []
    labels = []
    for i in range(len(data) - timestep):
        seq = data[i:i + timestep].values
        label = data['price'].iloc[i + timestep]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)


# 加载数据
df, cate_cols, num_cols, scaler = preprocess_data()
data = df[cate_cols + num_cols + ['price']]

# 创建时间序列数据集
X, y = create_sequences(data, config.timestep)
train_size = int(len(X) * 0.8)
x_train, y_train = X[:train_size], y[:train_size]
x_test, y_test = X[train_size:], y[train_size:]

# 转换为Tensor
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 创建DataLoader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size)


# 定义模型
class CNN_LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=100, embedding_dim=config.embedding_dim)

        # CNN模块
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=config.timestep, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # LSTM模块
        self.lstm = nn.LSTM(
            input_size=len(cate_cols) * config.embedding_dim + len(num_cols),
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, config.output_dim)
        )

    def forward(self, x):
        # 分离类别特征和数值特征
        cate = x[:, :, :len(cate_cols)].long()
        numeric = x[:, :, len(cate_cols):-1]

        # 嵌入层处理类别特征
        embedded = [self.embedding(cate[:, :, i]) for i in range(cate.shape[2])]
        embedded = torch.cat(embedded, dim=-1)

        # 合并特征
        combined = torch.cat([embedded, numeric], dim=-1)

        # CNN处理
        cnn_out = self.cnn(combined.permute(0, 2, 1))
        cnn_out = cnn_out.permute(0, 2, 1)

        # LSTM处理
        lstm_out, _ = self.lstm(cnn_out)

        # 全连接层
        output = self.fc(lstm_out[:, -1, :])
        return output


# 初始化模型
model = CNN_LSTM(config)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 训练过程
best_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(config.epochs):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.epochs}'):
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            val_loss += criterion(outputs, batch_y).item()

    # 记录损失
    avg_train_loss = epoch_loss / len(train_loader)
    avg_val_loss = val_loss / len(test_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # 保存最佳模型
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), config.save_path)

    print(f'Epoch {epoch + 1}/{config.epochs}')
    print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 模型评估
model.load_state_dict(torch.load(config.save_path))
model.eval()
with torch.no_grad():
    test_preds = model(x_test_tensor).numpy()
    test_true = y_test_tensor.numpy()

# 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(test_true[:100], label='True Price')
plt.plot(test_preds[:100], label='Predicted Price')
plt.title('Price Prediction Comparison')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()