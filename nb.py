# 导入所需库
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

# 读取数据集
train_data = pd.read_csv('traindata.csv', header=None)
test_data = pd.read_csv('testdata-21.csv', header=None)
print(train_data.shape)
print(test_data.shape)

# # 特征选择1
# train_data = train_data.iloc[:,[0,7,9,10,15,16,19,11,22,27,28,30,41]]
# test_data = test_data.iloc[:,[0,7,9,10,15,16,19,11,22,27,28,30,41]]

# 特征选择2
train_data = train_data.iloc[:,[2,3,4,28,38,41]]
test_data = test_data.iloc[:,[2,3,4,28,38,41]]

# 数据集划分
x_train = train_data.iloc[1:, :-1].astype('float')
y_train = train_data.iloc[1:, -1].astype('float')

x_test = test_data.iloc[1:,:-1].astype('float')
y_test = test_data.iloc[1:,-1].astype('float')


# # 调参
# # 创建参数空间
# param_grid = {
#     'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
# }
#
# # 网格搜索
# NB=GaussianNB()
# grid_search = GridSearchCV(estimator=NB, param_grid=param_grid, cv=5)
# grid_search.fit(x_train, y_train)
#
# # 输出最佳参数
# print("Best parameters: ", grid_search.best_params_)


# 朴素贝叶斯
# # test21 特征1
# NB=GaussianNB(var_smoothing= 1e-06)
# # test21 特征2
NB=GaussianNB(var_smoothing= 1e-06)

# NB=GaussianNB(var_smoothing= 1e-06)
start_time = time.time()
NB.fit(x_train, y_train)
end_time = time.time()
train_time = end_time - start_time
print(f"NB training time: {train_time:.4f} seconds")

start_time = time.time()
y_pred = NB.predict(x_test)
end_time = time.time()
test_time = end_time - start_time
print(f"NB testing time: {test_time:.4f} seconds")
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print('Accuracy:', accuracy,"with zero_division=True")
print('Classification Report:\n', report,"with zero_division=True")

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm,"with zero_division=True")
