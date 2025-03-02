import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import math


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

# raw_data['历史行为均值'] = raw_data[config['history_cols']].mean()
#
# # 计算标准差特征
# raw_data['历史行为标准差'] = raw_data[config['history_cols']].std()
#
# # # 计算最大值特征
# # raw_data['历史行为最大值'] = raw_data[config['history_cols']].max()
#
# # # 可选的特征转换
# # raw_data['历史行为指数'] = raw_data[config['history_cols']].apply(lambda x: math.exp(x))
# raw_data.to_csv('./look.csv')
#
# # 可选的特征选择
# selected_features = ['历史行为均值', '历史行为标准差']


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

# # 堆起来处理one-hot特征
# train_udmap_df = pd.DataFrame(np.vstack(train_data['udmap']))
# test_udmap_df = pd.DataFrame(np.vstack(test_data['udmap']))

# 分离训练集和测试集
train_data = raw_data[:557658]
test_data = raw_data[557658:]
# 打乱训练集
train_data = train_data.sample(frac=1)

test_data.to_csv('./ori_get_test.csv')

# 丢弃无用特征
train_data.drop(config['ignore'], axis=1, inplace=True)
test_data.drop(config['ignore'], axis=1, inplace=True)
# test_data['新5G终端品牌'] = test_data['新5G终端品牌'].fillna(-1)
test_data = test_data.drop(['新5G终端品牌'], axis=1)

# 使用LabelEncoder
train_data.to_csv('./wtf.csv')
enc = LabelEncoder()
enc = enc.fit(['一加', '华为', 'OPPO', 'vivo', '小米', '荣耀', '三星', '魅族', '其他', '中兴', '中国移动'])
train_data['新5G终端品牌'] = enc.transform(train_data['新5G终端品牌'])
train_data.to_csv("./aefaestgQ.csv")
# train_data['新5G终端品牌'] = enc.fit_transform(train_data['新5G终端品牌'].values.reshape(-1, 1)).toarray()

## 决策树模型
# clf = DecisionTreeClassifier(criterion="gini", random_state=0, splitter='best')
# clf.fit(
#     train_data.drop(['新5G终端品牌'], axis=1),
#     train_data['新5G终端品牌'],
# )
# score = clf.score(train_data.drop(['新5G终端品牌'], axis=1), train_data['新5G终端品牌'])

# XGBoost模型
clf = xgb.XGBClassifier(objective='multi:softmax', random_state=0, n_jobs=-1, eta=0.2, max_depth=7, subsample=0.8)
clf.fit(
    train_data.drop(['新5G终端品牌'], axis=1),
    train_data['新5G终端品牌'],
    eval_metric='auc'
)
score = clf.score(train_data.drop(['新5G终端品牌'], axis=1), train_data['新5G终端品牌'])
print(score)

# test = []
# for i in range(200):
#     clf = DecisionTreeClassifier(criterion="entropy", random_state=0, splitter='best')
#     clf = clf.fit(train_data.drop(['新5G终端品牌'], axis=1), train_data['新5G终端品牌'])
#     score = clf.score(train_data.drop(['新5G终端品牌'], axis=1), train_data['新5G终端品牌'])
#     test.append(score)
# plt.plot(range(1, 201), test, 'r')
# plt.legend()
# plt.show()


# to csv
ori = pd.read_csv('./ori_get_test.csv')
pred = clf.predict(test_data)
# result_df.to_csv('./result_df.csv', index=None)

values = ['OPPO', 'vivo', '一加', '三星', '中国移动', '中兴', '其他', '华为', '小米', '荣耀', '魅族']
pred = pred.tolist()

for i in range(len(pred)):
    pred[i] = values[pred[i]]

result_df = pd.DataFrame({
    '用户标识': ori['用户标识'],  # 使用测试数据集中的 'uuid' 列作为 'uuid' 列的值
    '新5G终端品牌': pred  # 使用模型 clf 对测试数据集进行预测，并将预测结果存储在 'target' 列中
})
print(result_df[result_df.isnull().T.any()])
result_df.to_csv('./result_df4.csv', index=None)




# # 用于映射手机品牌
# affine = dict()
# keys = [i for i in range(11)]
#
# for key in keys:
#     affine[key] = config['res'][key][8:]
#
# pred = clf.predict(test_data.drop([i for i in config['num_cols']] + [i for i in config['res']], axis=1))
# pred = np.argmax(pred, -1)
# pred = list(pred.detach().numpy())
#
# # 将索引换成手机品牌
# for index, value in enumerate(pred):
#     for key in affine:
#         if value == key:
#             pred[index] = affine[key]
#
# res = pd.DataFrame(pred)
# res.columns = ['新5G终端品牌']
# ori = pd.read_csv('../dl_code/ori_get_test.csv')
# res = pd.concat([ori['用户标识'], res], axis=1)
# print(res)
# # res.to_csv('result.csv', index=False)