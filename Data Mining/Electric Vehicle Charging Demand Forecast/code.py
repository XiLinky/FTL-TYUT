import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import h3

warnings.filterwarnings('ignore')

train_power_forecast_history = pd.read_csv('./训练集/power_forecast_history.csv')  # 站点运营数据
train_power = pd.read_csv('./训练集/power.csv')  # 站点充电量数据
train_stub_info = pd.read_csv('./训练集/stub_info.csv')  # 站点静态数据

test_power_forecast_history = pd.read_csv('./测试集/power_forecast_history.csv')
test_stub_info = pd.read_csv('./测试集/stub_info.csv')


# h3解码
for i in train_stub_info['h3']:
    longitude_train, latitude_train = h3.h3_to_geo(i)
    train_stub_info['longitude'] = longitude_train
    train_stub_info['latitude'] = latitude_train

for i in test_stub_info['h3']:
    longitude_test, latitude_test = h3.h3_to_geo(i)
    test_stub_info['longitude'] = longitude_test
    test_stub_info['latitude'] = latitude_test

del train_stub_info['h3']
del test_stub_info['h3']

train_stub_info.to_csv('train_stub_info_h3decode.csv')
test_stub_info.to_csv('test_stub_info_h3decode.csv')

# 聚合数据
train_df = train_power_forecast_history.groupby(['id_encode', 'ds']).head(1)  # 确保每个充电桩每天只有一个样本
del train_df['hour']  # 预估目标以天为单位，不需要hour

test_df = test_power_forecast_history.groupby(['id_encode', 'ds']).head(1)
del test_df['hour']

tmp_df = train_power.groupby(['id_encode', 'ds'])['power'].sum()  # 计算power总和，变Series对象
tmp_df.columns = ['id_encode', 'ds', 'power']

# 合并充电量数据
train_df = train_df.merge(tmp_df, on=['id_encode', 'ds'], how='left')

# 合并数据
train_df = train_df.merge(train_stub_info, on='id_encode', how='left')
test_df = test_df.merge(test_stub_info, on='id_encode', how='left')

# 定义要绘制的列
cols = ['power']

# 遍历id_encode的五个值
for ie in [0, 1, 2, 3, 4]:
    # 获取train_df中id_encode为当前值ie的所有行，并重置索引
    tmp_df = train_df[train_df['id_encode'] == ie].reset_index(drop=True)

    # 再次重置索引，并为新索引添加一个名为'index'的列
    tmp_df = tmp_df.reset_index(drop=True).reset_index()

    # 遍历要绘制的列
    for num, col in enumerate(cols):
        # 设置图的大小
        plt.figure(figsize=(20, 10))

        # 创建子图，总共有4行1列，当前为第num+1个子图
        plt.subplot(4, 1, num + 1)

        # 绘制图形：x轴为'index'，y轴为当前列的值
        plt.plot(tmp_df['index'], tmp_df[col])

        # 为当前子图设置标题，标题为当前列的名称
        plt.title(col)

# 显示图形
plt.show()

# 创建一个新的图，大小为20x5
plt.figure(figsize=(20, 5))

# 数据预处理
train_df['flag'] = train_df['flag'].map({'A': 0, 'B': 1})
test_df['flag'] = test_df['flag'].map({'A': 0, 'B': 1})

# train_df.head()

def get_time_feature(df, col):
    df_copy = df.copy()
    prefix = col + "_"
    df_copy['new_' + col] = df_copy[col]

    col = 'new_' + col
    df_copy[col] = pd.to_datetime(df_copy[col], format='%Y%m%d')
    print(df_copy[col])
    df_copy[prefix + 'year'] = df_copy[col].dt.year
    df_copy[prefix + 'month'] = df_copy[col].dt.month
    df_copy[prefix + 'day'] = df_copy[col].dt.day
    # df_copy[prefix + 'weekofyear'] = df_copy[col].dt.weekofyear  # 加上会掉点
    df_copy[prefix + 'dayofweek'] = df_copy[col].dt.dayofweek
    df_copy[prefix + 'is_wknd'] = df_copy[col].dt.dayofweek // 6
    df_copy[prefix + 'quarter'] = df_copy[col].dt.quarter
    df_copy[prefix + 'is_month_start'] = df_copy[col].dt.is_month_start.astype(int)
    df_copy[prefix + 'is_month_end'] = df_copy[col].dt.is_month_end.astype(int)
    del df_copy[col]

    return df_copy


train_df = get_time_feature(train_df, 'ds')
test_df = get_time_feature(test_df, 'ds')

cols = [f for f in test_df.columns if f not in ['ds', 'power', 'h3']]


# 模型训练与验证

# 使用K折交叉验证训练和验证模型
def cv_model(clf, train_x, train_y, test_x, seed=2023):
    # 定义折数并初始化KFold
    folds = 8
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    # 初始化oof预测和测试集预测
    oof = np.zeros(train_x.shape[0])
    test_predict = np.zeros(test_x.shape[0])
    cv_scores = []

    # KFold交叉验证
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i + 1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], \
            train_y[valid_index]

        # 转换数据为lightgbm数据格式
        train_matrix = clf.Dataset(trn_x, label=trn_y)
        valid_matrix = clf.Dataset(val_x, label=val_y)

        # 定义lightgbm参数
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'min_child_weight': 5,
            'num_leaves': 2 ** 7,
            'lambda_l2': 10,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 4,
            'learning_rate': 0.03,
            'seed': 2023,
            'nthread': 16,
            'verbose': -1,
            # 'device':'gpu'
        }

        # 训练模型
        model = clf.train(params, train_matrix, 3000, valid_sets=[train_matrix, valid_matrix], categorical_feature=[])

        # 获取验证和测试集的预测值
        val_pred = model.predict(val_x, num_iteration=model.best_iteration)
        test_pred = model.predict(test_x, num_iteration=model.best_iteration)

        oof[valid_index] = val_pred
        test_predict += test_pred / kf.n_splits

        # 计算并打印当前折的分数
        score = np.sqrt(mean_squared_error(val_pred, val_y))
        cv_scores.append(score)
        print(cv_scores)

    return oof, test_predict


# 调用上面的函数进行模型训练和预测
lgb_oof, lgb_test = cv_model(lgb, train_df[cols], train_df['power'], test_df[cols])

test_df['power'] = lgb_test
test_df['power'] = test_df['power'].apply(lambda x: 0 if x < 0 else x)
test_df[['id_encode', 'ds', 'power']].to_csv('result1.csv', index=False)