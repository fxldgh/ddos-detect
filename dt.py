# 导入所需库
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import time

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

# # 网格搜索调参
# tree = DecisionTreeClassifier()
#
# params = {
#     'max_depth': [3, 5, 7],
#     'min_samples_leaf': [1, 3, 5],
#     'criterion': ['gini', 'entropy']
# }
#
# gridsearch = GridSearchCV(estimator=tree, param_grid=params, cv=5, scoring='f1_macro')
# gridsearch.fit(x_train, y_train)
#
# print(gridsearch.best_params_)
# print(gridsearch.best_score_)


# # 交叉验证调参
# best_score = 0
# best_parameters = {}
# for depth in range(3, 8):
#     for leaf in range(1, 6, 2):
#         for criterion in ['gini', 'entropy']:
#             tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, criterion=criterion)
#             f1_scores = cross_val_score(tree, x_train, y_train, cv=5, scoring='f1_macro')
#             avg_f1_score = f1_scores.mean()
#
#             if avg_f1_score > best_score:
#                 best_score = avg_f1_score
#                 best_parameters = {'depth': depth, 'leaf': leaf, 'criterion': criterion}
#
#             print(f"depth:{depth},leaf:{leaf},criterion:{criterion},f1_macro:{avg_f1_score}")
#
# print(f"Best Parameters: {best_parameters}")
# print(f"Best Score: {best_score}")



# dt
# test
# 特征1
# DT = DecisionTreeClassifier(max_depth=7, criterion='gini', min_samples_leaf=1, random_state=1)
# # 特征2
# DT = DecisionTreeClassifier(max_depth=7, criterion='entropy', min_samples_leaf=1, random_state=1)

# # test21
# 特征1
# DT = DecisionTreeClassifier(max_depth=7, criterion='gini', min_samples_leaf=1, random_state=1)
# 特征2
DT = DecisionTreeClassifier(max_depth=7, criterion='entropy', min_samples_leaf=1, random_state=1)
start_time = time.time()
DT.fit(x_train, y_train)
end_time = time.time()
train_time = end_time - start_time
print(f"dt training time: {train_time:.4f} seconds")


start_time = time.time()
y_pred = DT.predict(x_test)
end_time = time.time()
test_time = end_time - start_time
print(f"dt testing time: {test_time:.4f} seconds")
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("dt")
print('Accuracy:', accuracy,"with zero_division=True")
print('Classification Report:\n', report,"with zero_division=True")

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm,"with zero_division=True")