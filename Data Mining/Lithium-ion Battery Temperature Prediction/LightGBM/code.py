# 导入所需的库
import pandas as pd  # 用于处理数据的工具
import lightgbm as lgb  # 机器学习模型 LightGBM
from sklearn.metrics import mean_absolute_error  # 评分 MAE 的计算函数
from sklearn.model_selection import train_test_split  # 拆分训练集与验证集工具
from tqdm import tqdm  # 显示循环的进度条工具
import os
from xgboost import XGBRegressor
# 数据准备
train = pd.read_csv("../锂离子电池生产参数调控及生产温度预测挑战赛公开数据-初赛/锂离子电池生产参数调控及生产温度预测挑战赛公开数据-初赛/train.csv")  # 原始训练数据。
test = pd.read_csv("../锂离子电池生产参数调控及生产温度预测挑战赛公开数据-初赛/锂离子电池生产参数调控及生产温度预测挑战赛公开数据-初赛/testzu.csv")  # 原始测试数据（用于提交）。

# 交叉特征
for i in range(1, 18):
    train[f'流量{i}/上部温度设定{i}'] = train[f'流量{i}'] / train[f'上部温度设定{i}']
    test[f'流量{i}/上部温度设定{i}'] = test[f'流量{i}'] / test[f'上部温度设定{i}']

    train[f'流量{i}/下部温度设定{i}'] = train[f'流量{i}'] / train[f'下部温度设定{i}']
    test[f'流量{i}/下部温度设定{i}'] = test[f'流量{i}'] / test[f'下部温度设定{i}']

    train[f'上部温度设定{i}/下部温度设定{i}'] = train[f'上部温度设定{i}'] / train[f'下部温度设定{i}']
    test[f'上部温度设定{i}/下部温度设定{i}'] = test[f'上部温度设定{i}'] / test[f'下部温度设定{i}']

# 历史平移
for i in range(1, 18):
    train[f'last1_流量{i}'] = train[f'流量{i}'].shift(1)
    train[f'last1_上部温度设定{i}'] = train[f'上部温度设定{i}'].shift(1)
    train[f'last1_下部温度设定{i}'] = train[f'下部温度设定{i}'].shift(1)

    test[f'last1_流量{i}'] = test[f'流量{i}'].shift(1)
    test[f'last1_上部温度设定{i}'] = test[f'上部温度设定{i}'].shift(1)
    test[f'last1_下部温度设定{i}'] = test[f'下部温度设定{i}'].shift(1)

# 差分特征
for i in range(1, 18):
    train[f'last1_diff_流量{i}'] = train[f'流量{i}'].diff(1)
    train[f'last1_diff_上部温度设定{i}'] = train[f'上部温度设定{i}'].diff(1)
    train[f'last1_diff_下部温度设定{i}'] = train[f'下部温度设定{i}'].diff(1)

    test[f'last1_diff_流量{i}'] = test[f'流量{i}'].diff(1)
    test[f'last1_diff_上部温度设定{i}'] = test[f'上部温度设定{i}'].diff(1)
    test[f'last1_diff_下部温度设定{i}'] = test[f'下部温度设定{i}'].diff(1)

# 窗口统计
for i in range(1, 18):
    train[f'win3_mean_流量{i}'] = (train[f'流量{i}'].shift(1) + train[f'流量{i}'].shift(2) + train[f'流量{i}'].shift(3)) / 3
    train[f'win3_mean_上部温度设定{i}'] = (train[f'上部温度设定{i}'].shift(1) + train[f'上部温度设定{i}'].shift(2) + train[
        f'上部温度设定{i}'].shift(3)) / 3
    train[f'win3_mean_下部温度设定{i}'] = (train[f'下部温度设定{i}'].shift(1) + train[f'下部温度设定{i}'].shift(2) + train[
        f'下部温度设定{i}'].shift(3)) / 3

    test[f'win3_mean_流量{i}'] = (test[f'流量{i}'].shift(1) + test[f'流量{i}'].shift(2) + test[f'流量{i}'].shift(3)) / 3
    test[f'win3_mean_上部温度设定{i}'] = (test[f'上部温度设定{i}'].shift(1) + test[f'上部温度设定{i}'].shift(2) + test[
        f'上部温度设定{i}'].shift(3)) / 3
    test[f'win3_mean_下部温度设定{i}'] = (test[f'下部温度设定{i}'].shift(1) + test[f'下部温度设定{i}'].shift(2) + test[
        f'下部温度设定{i}'].shift(3)) / 3

# 差分特征 + 历史平移
for i in range(1, 18):
    train[f'shift_diff_流量{i}'] = train[f'last1_流量{i}'] + train[f'last1_diff_流量{i}']
    train[f'shift_diff_上部温度设定{i}'] = train[f'last1_上部温度设定{i}'] + train[f'last1_diff_上部温度设定{i}']
    train[f'shift_diff_下部温度设定{i}'] = train[f'last1_下部温度设定{i}'] + train[f'last1_diff_下部温度设定{i}']

    test[f'shift_diff_流量{i}'] = test[f'last1_流量{i}'] + test[f'last1_diff_流量{i}']
    test[f'shift_diff_上部温度设定{i}'] = test[f'last1_上部温度设定{i}'] + test[f'last1_diff_上部温度设定{i}']
    test[f'shift_diff_下部温度设定{i}'] = test[f'last1_下部温度设定{i}'] + test[f'last1_diff_下部温度设定{i}']

submit = pd.DataFrame()  # 定义提交的最终数据。
submit["序号"] = test["序号"]  # 对齐测试数据的序号。

# 模型训练
pred_labels = [ '上部温度1', '上部温度2',
       '上部温度3', '上部温度4', '上部温度5', '上部温度6', '上部温度7', '上部温度8', '上部温度9', '上部温度10',
       '上部温度11', '上部温度12', '上部温度13', '上部温度14', '上部温度15', '上部温度16', '上部温度17',
       '下部温度1', '下部温度2', '下部温度3', '下部温度4', '下部温度5', '下部温度6', '下部温度7', '下部温度8',
       '下部温度9', '下部温度10', '下部温度11', '下部温度12', '下部温度13', '下部温度14', '下部温度15',
       '下部温度16', '下部温度17']  # 需要预测的标签。
train_set, valid_set = train_test_split(train, test_size=0.2)  # 拆分数据集。

# 设定 LightGBM 训练参，查阅参数意义：https://lightgbm.readthedocs.io/en/latest/Parameters.html
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'min_child_weight': 5,
    'num_leaves': 2 ** 5,
    'lambda_l2': 10,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 4,
    'learning_rate': 0.05,
    'seed': 2023,
    'nthread': 16,
    'verbose': -1,
}

no_info = lgb.callback.log_evaluation(period=-1)  # 禁用训练日志输出。


def time_feature(data: pd.DataFrame, pred_labels: list = None) -> pd.DataFrame:
    """提取数据中的时间特征。

    输入:
        data: Pandas.DataFrame
            需要提取时间特征的数据。

        pred_labels: list, 默认值: None
            需要预测的标签的列表。如果是测试集，不需要填入。

    输出: data: Pandas.DataFrame
            提取时间特征后的数据。
    """

    data = data.copy()  # 复制数据，避免后续影响原始数据。
    data = data.drop(columns=["序号"])  # 去掉”序号“特征。

    data["时间"] = pd.to_datetime(data["时间"])  # 将”时间“特征的文本内容转换为 Pandas 可处理的格式。
    data["month"] = data["时间"].dt.month  # 添加新特征“month”，代表”当前月份“。
    data["day"] = data["时间"].dt.day  # 添加新特征“day”，代表”当前日期“。
    data["hour"] = data["时间"].dt.hour  # 添加新特征“hour”，代表”当前小时“。
    data["minute"] = data["时间"].dt.minute  # 添加新特征“minute”，代表”当前分钟“。
    data["weekofyear"] = data["时间"].dt.isocalendar().week.astype(
        int)  # 添加新特征“weekofyear”，代表”当年第几周“，并转换成 int，否则 LightGBM 无法处理。
    data["dayofyear"] = data["时间"].dt.dayofyear  # 添加新特征“dayofyear”，代表”当年第几日“。
    data["dayofweek"] = data["时间"].dt.dayofweek  # 添加新特征“dayofweek”，代表”当周第几日“。
    data["is_weekend"] = data["时间"].dt.dayofweek // 6  # 添加新特征“is_weekend”，代表”是否是周末“，1 代表是周末，0 代表不是周末。

    data = data.drop(columns=["时间"])  # LightGBM 无法处理这个特征，它已体现在其他特征中，故丢弃。

    if pred_labels:  # 如果提供了 pred_labels 参数，则执行该代码块。
        data = data.drop(columns=[*pred_labels])  # 去掉所有待预测的标签。

    return data  # 返回最后处理的数据。


test_features = time_feature(test)  # 处理测试集的时间特征，无需 pred_labels。


MAE_scores = dict()  # 定义评分项。
# 从所有待预测特征中依次取出标签进行训练与预测。
kkk = 0
for pred_label in tqdm(pred_labels):
    train_features = time_feature(train_set, pred_labels=pred_labels)  # 处理训练集的时间特征。
    train_labels = train_set[pred_label]  # 训练集的标签数据。

    rf = model = XGBRegressor(objective='reg:squarederror', random_state=42, learning_rate=0.03, max_depth=15, n_estimators=400)
    rf.fit(train_features, train_labels)
    feature_importances = pd.Series(rf.feature_importances_, index=train_features.columns)
    top_features = feature_importances.nlargest(150).index
    train_features = train_features[top_features]

    train_data = lgb.Dataset(train_features, label=train_labels)  # 将训练集转换为 LightGBM 可处理的类型。

    valid_features = time_feature(valid_set, pred_labels=pred_labels)  # 处理验证集的时间特征。
    valid_features = valid_features[top_features]
    valid_labels = valid_set[pred_label]  # 验证集的标签数据。
    valid_data = lgb.Dataset(valid_features, label=valid_labels)  # 将验证集转换为 LightGBM 可处理的类型。

    # 训练模型，参数依次为：导入模型设定参数、导入训练集、导入验证集、禁止输出日志
    model = lgb.train(lgb_params, train_data, 200, valid_sets=valid_data, callbacks=[no_info])

    test_features_ = test_features[top_features]

    valid_pred = model.predict(valid_features, num_iteration=model.best_iteration)  # 选择效果最好的模型进行验证集预测。
    test_pred = model.predict(test_features_, num_iteration=model.best_iteration)  # 选择效果最好的模型进行测试集预测。
    MAE_score = mean_absolute_error(valid_pred, valid_labels)  # 计算验证集预测数据与真实数据的 MAE。
    MAE_scores[pred_label] = MAE_score  # 将对应标签的 MAE 值 存入评分项中。

    submit[pred_label] = test_pred  # 将测试集预测数据存入最终提交数据中。
    kkk += 1
    print(kkk)
    print(MAE_scores)
# 创建保存模型的文件夹
os.makedirs('result', exist_ok=True)

# submit.to_csv('result/submit_result_mul_.csv', index=False)  # 保存最后的预测结果到 submit_result.csv。
submit.to_csv('submit_result_mul_shift+diff.csv', index=False)  # 保存最后的预测结果到 submit_result.csv。
print(MAE_scores)  # 查看各项的 MAE 值。