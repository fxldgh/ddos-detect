from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# 读取数据集
train_data = pd.read_csv('traindata.csv', header=None)
test_data = pd.read_csv('testdata-21.csv', header=None)

# # 特征选择1
# train_data = train_data.iloc[:,[0,7,9,10,15,16,19,11,22,27,28,30,41]]
# test_data = test_data.iloc[:,[0,7,9,10,15,16,19,11,22,27,28,30,41]]

# 特征选择2
train_data = train_data.iloc[:,[2,3,4,28,38,41]]
test_data = test_data.iloc[:,[2,3,4,28,38,41]]

x_train = train_data.iloc[1:, :-1].astype('float')
y_train = train_data.iloc[1:, -1].astype('float')

x_test = test_data.iloc[1:,:-1].astype('float')
y_test = test_data.iloc[1:,-1].astype('float')


# GridSearchCV
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
score = rf.score(x_test, y_test)
cross_s = cross_val_score(rf, x_test, y_test, cv=5).mean()
print('rf:', score)
print('cv:', cross_s)

# 调参第一步：n_estimators
cross = []
for i in range(0, 200, 10):
    rf = RandomForestClassifier(n_estimators=i + 1, n_jobs=-1, random_state=42)
    cross_score = cross_val_score(rf, x_test, y_test, cv=5).mean()
    cross.append(cross_score)
plt.plot(range(1, 201, 10), cross)
plt.xlabel('n_estimators')
plt.ylabel('acc')
# plt.savefig('rf_test211_n_estimators1.png')
plt.show()
# print((cross.index(max(cross)) * 10) + 1, max(cross))

# # n_estimators缩小范围
# cross = []
# for i in range(0, 20):
#     rf = RandomForestClassifier(n_estimators=i + 1, n_jobs=-1, random_state=42)
#     cross_score = cross_val_score(rf, x_test, y_test, cv=5).mean()
#     cross.append(cross_score)
# plt.plot(range(1, 21), cross)
# plt.xlabel('n_estimators')
# plt.ylabel('acc')
# plt.savefig('rf_test211_n_estimators2.png')
# plt.show()
# print(cross.index(max(cross)) + 1, max(cross))
#
# # 调整max_depth
# param_grid = {'max_depth': np.arange(1, 20, 1)}
# # 一般根据数据大小进行尝试，像该数据集 可从1-10 或1-20开始
# rf = RandomForestClassifier(n_estimators=11, random_state=42)
# GS = GridSearchCV(rf, param_grid, cv=5)
# GS.fit(x_train, y_train)
# GS.best_params_
# GS.best_score_
# print("Best parameters:",GS.best_params_)
# print("Best score:",GS.best_score_)
#
#
# # 调整max_features
# param_grid = {'max_features': np.arange(5, 30, 1)}
# rf = RandomForestClassifier(n_estimators=11, random_state=42)
# GS = GridSearchCV(rf, param_grid, cv=5)
# GS.fit(x_train, y_train)
# GS.best_params_
# GS.best_score_
# print("Best parameters:",GS.best_params_)
# print("Best score:",GS.best_score_)
#
# # 调整min_samples_leaf
# param_grid = {'min_samples_leaf': np.arange(1, 1 + 10, 1)}
# # 一般是从其最小值开始向上增加10或者20
# # 面对高维度高样本数据，如果不放心，也可以直接+50，对于大型数据可能需要增加200-300
# # 如果调整的时候发现准确率怎么都上不来，那可以放心大胆调一个很大的数据，大力限制模型的复杂度
# rf = RandomForestClassifier(n_estimators=11, random_state=42)
# GS = GridSearchCV(rf, param_grid, cv=5)
# GS.fit(x_train, y_train)
# GS.best_params_
# GS.best_score_
# print("Best parameters:",GS.best_params_)
# print("Best score:",GS.best_score_)
#
# # 调整min_samples_split
# param_grid = {'min_samples_split': np.arange(2, 2 + 20, 1)}
# # 一般是从其最小值开始向上增加10或者20
# # 面对高维度高样本数据，如果不放心，也可以直接+50，对于大型数据可能需要增加200-300
# # 如果调整的时候发现准确率怎么都上不来，那可以放心大胆调一个很大的数据，大力限制模型的复杂度
# rf = RandomForestClassifier(n_estimators=11, random_state=42)
# GS = GridSearchCV(rf, param_grid, cv=5)
# GS.fit(x_train, y_train)
# GS.best_params_
# GS.best_score_
# print("Best parameters:",GS.best_params_)
# print("Best score:",GS.best_score_)
#
# # 调整criterion
# param_grid = {'criterion': ['gini', 'entropy']}
# # 一般是从其最小值开始向上增加10或者20
# # 面对高维度高样本数据，如果不放心，也可以直接+50，对于大型数据可能需要增加200-300
# # 如果调整的时候发现准确率怎么都上不来，那可以放心大胆调一个很大的数据，大力限制模型的复杂度
# rf = RandomForestClassifier(n_estimators=11, random_state=42)
# GS = GridSearchCV(rf, param_grid, cv=5)
# GS.fit(x_train, y_train)
# GS.best_params_
# GS.best_score_
# print("Best parameters:",GS.best_params_)
# print("Best score:",GS.best_score_)