from PIL import Image
from feature import NPDFeature
from ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np
import glob
import os
import random
import pickle

def loadDataSet():
    if not os.path.exists('tmp.pkl'):
        samples = []
        # 遍历数据集目录
        for img_name in sorted(glob.glob('datasets/original/face/*.jpg')):
            # 读取正样本图像
            print(img_name)
            img = np.array(Image.open(img_name).resize((24, 24)).convert("L"))
            # 提取NPD特征
            features = NPDFeature(img).extract()
            sample = np.r_[features, 1]  # 在正样本特征后面加一个Label为1
            samples.append(sample)
        for img_name in sorted(glob.glob('datasets/original/nonface/*.jpg')):
            # 读取负样本图像
            print(img_name)
            img = np.array(Image.open(img_name).resize((24, 24)).convert("L"))
            # 提取NPD特征
            features = NPDFeature(img).extract()
            sample = np.r_[features, -1]  # 在负样本特征后面加一个Label为-1
            samples.append(sample)
        # 数据集打乱
        random.shuffle(samples)
        dataset = np.array(samples)
        with open('tmp.pkl', 'wb') as output:
            pickle.dump(dataset, output, True)

    with open('tmp.pkl', 'rb') as input:
        dataset = pickle.load(input)
        print(dataset.shape)
    # 将数据集切分为训练集和验证集
    X_train = dataset[:dataset.shape[0] * 3 // 4, :dataset.shape[1] - 1]
    y_train = dataset[:dataset.shape[0] * 3 // 4, dataset.shape[1] - 1]
    X_validation = dataset[dataset.shape[0] * 3 // 4:, :dataset.shape[1] - 1]
    y_validation = dataset[dataset.shape[0] * 3 // 4:, dataset.shape[1] - 1]
    return X_train, X_validation, y_train, y_validation


if __name__ == "__main__":
    X_train, X_validation, y_train, y_validation = loadDataSet()
    abc = AdaBoostClassifier(DecisionTreeClassifier, 20)
    abc.fit(X_train, y_train)
    final_pre_y = abc.predict(X_validation)
    error = 0
    for i in range(final_pre_y.shape[0]):
        if final_pre_y[i] != y_validation[i]:
            error = error + 1
    accuracy = 1 - error / y_validation.shape[0]
    print('accuracy: %f' % accuracy)
    target_names = ['face', 'nonface']
    report = classification_report(y_validation, final_pre_y, target_names=target_names)
    print(report)
    with open('report.txt', 'w') as f:
        f.write(report)