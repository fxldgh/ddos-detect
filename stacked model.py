# 导入所需库
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import time

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 读取数据集
train_data = pd.read_csv('traindata.csv', header=None)
test_data = pd.read_csv('testdata.csv', header=None)
print(train_data.shape)
print(test_data.shape)

# # 特征选择1
# train_data = train_data.iloc[:,[0,7,9,10,15,16,19,11,22,27,28,30,41]]
# test_data = test_data.iloc[:,[0,7,9,10,15,16,19,11,22,27,28,30,41]]

# # 特征选择2
# train_data = train_data.iloc[:,[2,3,4,28,38,41]]
# test_data = test_data.iloc[:,[2,3,4,28,38,41]]

x_train = train_data.iloc[1:, :-1].astype('float')
y_train = train_data.iloc[1:, -1].astype('float')

x_test = test_data.iloc[1:,:-1].astype('float')
y_test = test_data.iloc[1:,-1].astype('float')



# 定义特征子集
features1 = [0,7,9,10,15,16,19,11,22,27,28,30]
features2 = [2,3,4,28,38]

# 获取特征子集
x_train1 = x_train.iloc[:,features1]
x_test1 = x_test.iloc[:,features1]
x_train2 = x_train.iloc[:,features2]
x_test2 = x_test.iloc[:,features2]


# 定义模型实例
# SVM
SVM = SVC(kernel='linear', random_state=0)

# 朴素贝叶斯
NB=GaussianNB()

# knn
# 特征2
KNN2 = KNeighborsClassifier(n_neighbors=9,weights='distance')

# RF
# 特征1
RF1 = RandomForestClassifier(n_estimators = 93, max_features = 5, min_samples_leaf = 8, min_samples_split = 21, criterion = 'gini', random_state = 0)


# DT
# 特征1
DT1 = DecisionTreeClassifier(max_depth=7, criterion='gini', min_samples_leaf=1, random_state=1)


# 定义Pipeline实例：将特征子集和模型包装到一起
SVM1_pipe = Pipeline([('clf', SVM)])
SVM2_pipe = Pipeline([('clf', SVM)])
KNN2_pipe = Pipeline([('clf', KNN2)])
DT1_pipe = Pipeline([('clf',DT1)])
RF1_pipe = Pipeline([('clf',RF1)])

# 拟合模型
SVM1_pipe.fit(x_train1, y_train)
SVM2_pipe.fit(x_train2, y_train)
KNN2_pipe.fit(x_train2, y_train)
DT1_pipe.fit(x_train1,y_train)
RF1_pipe.fit(x_train1,y_train)


#
estimator_list = [
    ('SVM1',SVM1_pipe),
    ('SVM2',SVM2_pipe),
    ('KNN2',KNN2_pipe),
    ('DT1',DT1_pipe)
    ]

# Build and fit stack model
stack_model = StackingClassifier(
    estimators=estimator_list, final_estimator=LogisticRegression(),cv=5,n_jobs=-1,passthrough=False,verbose=2)
start_time = time.time()
stack_model.fit(x_train, y_train)
end_time = time.time()
train_time = end_time - start_time
print(f"model training time: {train_time:.4f} seconds")

# 保存模型
dump(stack_model, '../Model_load/stack_test.dat')


# start_time = time.time()
# y_pred = stack_model.predict(x_test)
# end_time = time.time()
# test_time = end_time - start_time
# print(f"model testing time: {test_time:.4f} seconds")
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)
# print("stacked model")
# print('Accuracy:', accuracy,"with zero_division=True")
# print('Classification Report:\n', report,"with zero_division=True")
#
# # Evaluate the model
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", cm,"with zero_division=True")
