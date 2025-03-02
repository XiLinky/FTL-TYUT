import os

import pandas as pd
import torch

config = {
    'lr': 3e-2,
    'batch_size': 64,
    'epoch': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'res': ['新5G终端品牌_一加', '新5G终端品牌_华为', '新5G终端品牌_OPPO', '新5G终端品牌_vivo', '新5G终端品牌_小米',
            '新5G终端品牌_荣耀', '新5G终端品牌_三星', '新5G终端品牌_魅族', '新5G终端品牌_其他', '新5G终端品牌_中兴',
            '新5G终端品牌_中国移动'],
    'ignore': ['用户标识', '新5G终端品牌_-1', '标签月'],
    'cate_cols': ['用户状态', '性别', '是否4G客户', '客户类型', '城县农标志', '是否全球通客户', '上月是否集团统付',
                  '上月主套餐是否是大流量套餐', '当前终端是否合约终端', '当前终端品牌', '当前终端价格档次',
                  '当前终端是否双卡槽', '上次终端品牌', '上次终端价格档次', '上次终端是否双卡槽', '上上次终端品牌',
                  '上上次终端价格档次', '上上次终端是否双卡槽', '家庭交往圈终端品牌偏好', '用户历史终端品牌偏好',
                  '集团交往圈终端品牌偏好', '家庭交往圈是否有5G终端', '集团交往圈是否有5G终端', '是否订购5G套餐',
                  '用户常驻地是否在5G基站3公里范围', '是否资费敏感客户', '新5G终端档次', '新5G终端品牌'],
    # 'res': ['新5G终端品牌'],
    # 'ignore': ['用户标识', '标签月'],
    # 'cate_cols': ['用户状态', '性别', '是否4G客户', '客户类型', '城县农标志', '是否全球通客户', '上月是否集团统付',
    #               '上月主套餐是否是大流量套餐', '当前终端是否合约终端', '当前终端品牌', '当前终端价格档次',
    #               '当前终端是否双卡槽', '上次终端品牌', '上次终端价格档次', '上次终端是否双卡槽', '上上次终端品牌',
    #               '上上次终端价格档次', '上上次终端是否双卡槽', '家庭交往圈终端品牌偏好', '用户历史终端品牌偏好',
    #               '集团交往圈终端品牌偏好', '家庭交往圈是否有5G终端', '集团交往圈是否有5G终端', '是否订购5G套餐',
    #               '用户常驻地是否在5G基站3公里范围', '是否资费敏感客户', '新5G终端档次'],
    'num_cols': ['年龄', '网龄', '用户星级', '同身份证下的号码数', '近三年换终端次数', '上月主套餐费用',
                 '上月主套内包含流量', '上月ARPU', '上上月ARPU', '上上上月ARPU', '当前终端合约剩余期限',
                 '终端合约月最低消费', '用户换机时长', '历史终端使用平均时长', '当前终端价格', '当前终端使用时长',
                 '上次终端价格', '上次终端使用时长', '上上次终端价格', '上上次终端使用时长', '上月DOU', '上上月DOU',
                 '上上上月DOU', '上月流量饱和度', '上上月流量饱和度', '上上上月流量饱和度', '近3月月均流量超套费用',
                 '上月限速次数', '上月2', '近三月限速次数', '近三月2', '近3月月均上网天数', '上月游戏类流量',
                 '上月音乐类流量', '上月视频类流量', '上月短视频类流量', '上月游戏类流量占比', '上月音乐类流量占比',
                 '上月视频类流量占比', '上月短视频类流量占比']
}

raw_data = pd.read_csv(os.path.join('../testa/toUser_train_data.csv'))
test_data = pd.read_csv(os.path.join('../testa/test_A.csv'))
raw_data = pd.concat([raw_data, test_data])


# print(raw_data.shape)


# one-hot-encode处理字符串数据
def oneHotEncode(df, colNames):
    for col in colNames:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)
        df.drop([col], axis=1, inplace=True)
    return df


# 归一化处理数字数据
for col in config['num_cols']:
    raw_data[col] = raw_data[col].fillna(0.0)
    raw_data[col] = (raw_data[col] - raw_data[col].min()) / (raw_data[col].max() - raw_data[col].min())
    # print(raw_data[col])

# one-hot-encode处理离散数据
for col in config['cate_cols']:
    raw_data[col] = raw_data[col].fillna('-1')
raw_data = oneHotEncode(raw_data, config['cate_cols'])

# 分离训练集和测试集
train_data = raw_data[:557658]
test_data = raw_data[557658:]

test_data.to_csv('./ori_get_test.csv')

# 用户标识之后加
train_data.drop(config['ignore'], axis=1, inplace=True)
test_data.drop(config['ignore'], axis=1, inplace=True)
test_data.drop(config['res'], axis=1, inplace=True)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

train_target = train_data[config['res']]
train_target.to_csv('./train_target.csv')

train_data.drop(config['res'], axis=1, inplace=True)
train_data.to_csv('./train_data.csv')
test_data.to_csv('./test_feature.csv')

# 打乱
train_data = train_data.sample(frac=1)

# 1:100分离验证集
validation_feature = train_data[:5500]
train_features = train_data[5500:]
validation_target = train_target[:5500]
train_targets = train_target[5500:]

# 转为tensor
train_feature = torch.tensor(train_features.values, dtype=torch.float32, device=config['device'])
train_target = torch.tensor(train_targets.values, dtype=torch.float32, device=config['device'])

validation_feature = torch.tensor(validation_feature.values, dtype=torch.float32, device=config['device'])
validation_target = torch.tensor(validation_target.values, dtype=torch.float32, device=config['device'])

# delcol = ['新5G终端品牌_-1']
# data = test_data.drop(columns=delcol)
test_data = torch.tensor(test_data.values, dtype=torch.float32)
# print(test_data.shape)
