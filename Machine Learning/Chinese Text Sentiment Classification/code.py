import jieba
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle


class Config:
    file_path = './data_single.csv'  # 数据路径
    stop_words_path = './stopwords.txt'
    test_size = 0.1
    random_state = 2023
    max_features = 7200
    alpha = 1
    model_path = 'sentiment_model.pkl'
    dict_path = 'label_dict.pkl'


config = Config()


def load_data():
    """加载并预处理数据"""
    df = pd.read_csv(config.file_path)

    # 分词处理
    df['segmented'] = df['evaluation'].apply(
        lambda x: ' '.join(jieba.cut(x, cut_all=False))
    )

    # 加载停用词
    with open(config.stop_words_path, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])

    return df, stopwords


def prepare_features(df, stopwords):
    """特征工程"""
    # 文本向量化
    cv = CountVectorizer(
        stop_words=list(stopwords),
        max_features=config.max_features,
        token_pattern=r"(?u)\b\w+\b"  # 确保处理单字
    )
    X = cv.fit_transform(df['segmented']).toarray()

    # 保存词向量模型
    with open('count_vectorizer.pkl', 'wb') as f:
        pickle.dump(cv, f)

    # 处理标签
    labels = df['label'].unique()
    label_dict = {label: idx for idx, label in enumerate(labels)}
    y = df['label'].map(label_dict).values

    return X, y, label_dict


def train_model(X, y):
    """模型训练与评估"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state
    )

    model = MultinomialNB(alpha=config.alpha)
    model.fit(X_train, y_train)

    print(f'训练集准确率: {model.score(X_train, y_train):.4f}')
    print(f'测试集准确率: {model.score(X_test, y_test):.4f}')

    return model


def save_artifacts(model, label_dict):
    """保存模型和字典"""
    with open(config.model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(config.dict_path, 'wb') as f:
        pickle.dump(label_dict, f)


def predict_sentiment(sentence):
    """情感预测函数"""
    try:
        # 加载必要组件
        with open('count_vectorizer.pkl', 'rb') as f:
            cv = pickle.load(f)

        with open(config.model_path, 'rb') as f:
            model = pickle.load(f)

        with open(config.dict_path, 'rb') as f:
            label_dict = pickle.load(f)

        # 文本预处理
        seg_sentence = ' '.join(jieba.cut(sentence, cut_all=False))
        vec_sentence = cv.transform([seg_sentence]).toarray()

        # 预测
        pred = model.predict(vec_sentence)
        inverted_dict = {v: k for k, v in label_dict.items()}

        print(f'输入语句: {sentence}')
        print(f'情感预测结果: {inverted_dict[pred[0]]}')

    except Exception as e:
        print(f"预测出错: {str(e)}")
        print("请检查模型文件是否存在，或输入是否符合规范")


if __name__ == "__main__":
    # 数据加载
    df, stopwords = load_data()

    # 特征工程
    X, y, label_dict = prepare_features(df, stopwords)

    # 模型训练
    model = train_model(X, y)

    # 保存模型
    save_artifacts(model, label_dict)

    # 测试预测
    predict_sentiment("商品很满意，我还会再光临的")