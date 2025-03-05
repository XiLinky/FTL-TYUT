import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


class Config:
    dataset_name = 'MUTAG'
    hidden_channels = 64
    num_layers = 3
    dropout = 0.5
    lr = 0.01
    weight_decay = 1e-4
    epochs = 200
    patience = 20  # Early stopping
    batch_size = 64
    data_split = (0.8, 0.1, 0.1)  # Train/Val/Test


# 数据加载与预处理
dataset = TUDataset(root='data/TUDataset', name=Config.dataset_name)
torch.manual_seed(12345)
dataset = dataset.shuffle()

# 划分数据集
train_size = int(len(dataset) * Config.data_split[0])
val_size = int(len(dataset) * Config.data_split[1])
test_size = len(dataset) - train_size - val_size

train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:(train_size + val_size)]
test_dataset = dataset[(train_size + val_size):]

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)


# 定义改进后的GCN模型
class AdvancedGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, num_layers=3):
        super().__init__()
        self.conv_layers = torch.nn.ModuleList()
        self.dropout = Dropout(Config.dropout)

        # 输入层
        self.conv_layers.append(GCNConv(num_node_features, hidden_channels))

        # 隐藏层
        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))

        # 输出层
        self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))

        # 全连接层
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 多层GCN处理
        for conv in self.conv_layers[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        # 最后一层不加激活
        x = self.conv_layers[-1](x, edge_index)

        # 全局池化
        x = global_mean_pool(x, batch)
        x = self.dropout(x)

        # 分类层
        x = self.lin(x)
        return x


model = AdvancedGCN(
    num_node_features=dataset.num_node_features,
    hidden_channels=Config.hidden_channels,
    num_classes=dataset.num_classes,
    num_layers=Config.num_layers
)

print(model)

# 优化器和损失函数
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=Config.lr,
    weight_decay=Config.weight_decay
)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
criterion = torch.nn.CrossEntropyLoss()


# 早停机制
class EarlyStopper:
    def __init__(self, patience=10):
        self.best_val_loss = float('inf')
        self.patience = patience
        self.counter = 0

    def check(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return False
        return True


early_stopper = EarlyStopper(Config.patience)

# 训练和验证过程
train_losses = []
val_losses = []
train_accs = []
val_accs = []


def train():
    model.train()
    total_loss = 0
    correct = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 计算准确率
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / len(train_loader.dataset)
    train_losses.append(avg_loss)
    train_accs.append(accuracy)
    return avg_loss, accuracy


def evaluate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()

            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


# 主训练循环
best_test_acc = 0
for epoch in range(1, Config.epochs + 1):
    train_loss, train_acc = train()
    val_loss, val_acc = evaluate(val_loader)
    scheduler.step()

    # 记录验证损失和准确率
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # 早停检查
    if not early_stopper.check(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break

    # 评估测试集
    test_loss, test_acc = evaluate(test_loader)
    if test_acc > best_test_acc:
        best_test_acc = test_acc

    print(f'Epoch: {epoch:03d}, '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# 最终评估
model.eval()
test_acc = evaluate(test_loader)[1]
print(f'Final Test Accuracy: {test_acc:.4f}')

# 绘制学习曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Training Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()