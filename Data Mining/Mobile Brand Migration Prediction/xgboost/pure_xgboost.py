import pandas as pd
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# 配置参数

config = {
    'res': ['新5G终端品牌'],
    'ignore': ['用户标识', '标签月'],
    'cate_cols': ['用户状态', '性别', '是否4G客户', '客户类型', '城县农标志', '是否全球通客户', '上月是否集团统付',
                  '上月主套餐是否是大流量套餐', '当前终端是否合约终端', '当前终端品牌', '当前终端价格档次',
                  '当前终端是否双卡槽', '上次终端品牌', '上次终端价格档次', '上次终端是否双卡槽', '上上次终端品牌',
                  '上上次终端价格档次', '上上次终端是否双卡槽', '家庭交往圈终端品牌偏好', '用户历史终端品牌偏好',
                  '集团交往圈终端品牌偏好', '家庭交往圈是否有5G终端', '集团交往圈是否有5G终端', '是否订购5G套餐',
                  '用户常驻地是否在5G基站3公里范围', '是否资费敏感客户', '新5G终端档次'],
    'num_cols': ['年龄', '网龄', '用户星级', '同身份证下的号码数', '近三年换终端次数', '上月主套餐费用',
                 '上月主套内包含流量', '上月ARPU', '上上月ARPU', '上上上月ARPU', '当前终端合约剩余期限',
                 '终端合约月最低消费', '用户换机时长', '历史终端使用平均时长', '当前终端价格', '当前终端使用时长',
                 '上次终端价格', '上次终端使用时长', '上上次终端价格', '上上次终端使用时长', '上月DOU', '上上月DOU',
                 '上上上月DOU', '上月流量饱和度', '上上月流量饱和度', '上上上月流量饱和度', '近3月月均流量超套费用',
                 '上月限速次数', '上月2', '近三月限速次数', '近三月2', '近3月月均上网天数', '上月游戏类流量',
                 '上月音乐类流量', '上月视频类流量', '上月短视频类流量', '上月游戏类流量占比', '上月音乐类流量占比',
                 '上月视频类流量占比', '上月短视频类流量占比'],
    'history_cols': ['近三年换终端次数', '上月主套餐费用',
                 '上月主套内包含流量', '当前终端价格档次',
                  '当前终端是否双卡槽', '上次终端品牌', '上次终端价格档次', '上次终端是否双卡槽', '上上次终端品牌',
                  '上上次终端价格档次', '上上次终端是否双卡槽', '历史终端使用平均时长', '当前终端价格', '当前终端使用时长',
                 '上次终端价格', '上次终端使用时长', '上上次终端价格', '上上次终端使用时长','近3月月均上网天数', '上月游戏类流量',
                 '上月音乐类流量', '上月视频类流量', '上月短视频类流量', '上月游戏类流量占比', '上月音乐类流量占比',
                 '上月视频类流量占比', '上月短视频类流量占比']
}


# 读取数据
raw_data = pd.read_csv(os.path.join('../testa/toUser_train_data.csv'))
test_data = pd.read_csv(os.path.join('./testb/test_B.csv'))
raw_data = pd.concat([raw_data, test_data])


def oneHotEncode(df, colNames):
    for col in colNames:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)
        df.drop([col], axis=1, inplace=True)  # drop原数据
    return df


# 归一化处理数字数据
for col in config['num_cols']:
    raw_data[col] = raw_data[col].fillna(0.0)
    raw_data[col] = (raw_data[col] - raw_data[col].min()) / (raw_data[col].max() - raw_data[col].min())

# one-hot-encode处理离散数据
for col in config['cate_cols']:
    raw_data[col] = raw_data[col].fillna('-1')
raw_data = oneHotEncode(raw_data, config['cate_cols'])

# 分离训练集和测试集
train_data = raw_data[:557658]
test_data = raw_data[557658:]
# 打乱训练集
train_data = train_data.sample(frac=1)

# 用于获取用户标识
test_data.to_csv('./ori_get_test.csv')

# 丢弃无用特征
train_data.drop(config['ignore'], axis=1, inplace=True)
test_data.drop(config['ignore'], axis=1, inplace=True)
test_data = test_data.drop(['新5G终端品牌'], axis=1)

# 使用LabelEncoder
enc = LabelEncoder()
enc = enc.fit(['一加', '华为', 'OPPO', 'vivo', '小米', '荣耀', '三星', '魅族', '其他', '中兴', '中国移动'])
train_data['新5G终端品牌'] = enc.transform(train_data['新5G终端品牌'])

# XGBoost模型
clf = xgb.XGBClassifier(objective='multi:softmax', random_state=0, n_jobs=-1, )
clf.fit(
    train_data.drop(['新5G终端品牌'], axis=1),
    train_data['新5G终端品牌'],
    eval_metric='auc'
)
score = clf.score(train_data.drop(['新5G终端品牌'], axis=1), train_data['新5G终端品牌'])
print(score)

# predict
ori = pd.read_csv('./ori_get_test.csv')
pred = clf.predict(test_data)

values = ['OPPO', 'vivo', '一加', '三星', '中国移动', '中兴', '其他', '华为', '小米', '荣耀', '魅族']
pred = pred.tolist()

for i in range(len(pred)):
    pred[i] = values[pred[i]]

# to csv
result_df = pd.DataFrame({
    '用户标识': ori['用户标识'],
    '新5G终端品牌': pred
})
result_df.to_csv('./result_df.csv', index=None)