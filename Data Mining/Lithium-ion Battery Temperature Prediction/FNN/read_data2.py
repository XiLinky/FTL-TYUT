import pandas as pd
import torch
import os


config = {
    'lr': 3e-4,
    'batch_size': 64,
    'epoch': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'res': ['上部温度'+str(i+1) for i in range(17)] + ['下部温度'+str(i+1) for i in range(17)],
    'ignore': ['序号', '时间'],
    'cate_cols': [],
    'num_cols': ['上部温度设定'+str(i+1) for i in range(17)] + ['下部温度设定'+str(i+1) for i in range(17)] + ['流量'+str(i+1) for i in range(17)]
}

raw_data = pd.read_csv(os.path.join('../锂离子电池生产参数调控及生产温度预测挑战赛公开数据-初赛/锂离子电池生产参数调控及生产温度预测挑战赛公开数据-初赛/train.csv'))
test_data = pd.read_csv(os.path.join('../锂离子电池生产参数调控及生产温度预测挑战赛公开数据-初赛/锂离子电池生产参数调控及生产温度预测挑战赛公开数据-初赛/test.csv'))
raw_data = pd.concat([raw_data, test_data])


def oneHotEncode(df, colNames):
    for col in colNames:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)
        df.drop([col], axis=1, inplace=True)
    return df


# 处理离散数据
for col in config['cate_cols']:
    raw_data[col] = raw_data[col].fillna('-1')
raw_data = oneHotEncode(raw_data, config['cate_cols'])
# 处理连续数据
for col in config['num_cols']:
    raw_data[col] = raw_data[col].fillna(0)
    raw_data[col] = (raw_data[col]-raw_data[col].min()) / (raw_data[col].max()-raw_data[col].min())


raw_data.drop(config['ignore'], axis=1, inplace=True)


all_data = raw_data.astype('float32')


# 暂存处理后的test数据集
test_data = all_data[pd.isna(all_data['下部温度1'])]
print(test_data)
test_data.to_csv('./one_hot_test.csv')

train_data = all_data[pd.notna(all_data['下部温度1'])]
print(train_data)

# 打乱
train_data = train_data.sample(frac=1)

# 分离目标
train_target = train_data[config['res']]
train_data.drop(config['res'], axis=1, inplace=True)

train_target.to_csv('./train_target.csv')


# 分离出验证集，用于观察拟合情况
validation_data = train_data[:800]
train_data = train_data[800:]
validation_target = train_target[:800]
train_target = train_target[800:]

# 转为tensor
train_features = torch.tensor(train_data.values, dtype=torch.float32, device=config['device'])
train_num = train_features.shape[0]
train_labels = torch.tensor(train_target.values, dtype=torch.float32, device=config['device'])

validation_features = torch.tensor(validation_data.values, dtype=torch.float32, device=config['device'])
validation_num = validation_features.shape[0]
validation_labels = torch.tensor(validation_target.values, dtype=torch.float32, device=config['device'])

delcol = ['上部温度'+str(i+1) for i in range(17)] + ['下部温度'+str(i+1) for i in range(17)]
data = test_data.drop(columns=delcol)
test = torch.tensor(data.values, dtype=torch.float32)