# 导入所需库
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

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
# knn = KNeighborsClassifier()
# param_grid = {'n_neighbors': [3, 5, 7, 9],'weights': ['uniform', 'distance']}
# grid = GridSearchCV(knn, param_grid, cv=5)
# start_time = time.time()
# grid.fit(x_train, y_train)
# end_time = time.time()
# train_time = end_time - start_time
# print(f"knn training time: {train_time:.4f} seconds")
# best_params = grid.best_params_
# best_k = best_params['n_neighbors']
# best_weight = best_params['weights']
# print('Best hyperparameters: ', best_params)
# print('Best K value:', best_k)

# knn
# #test
# # 特征1
# # knn = KNeighborsClassifier(n_neighbors=9,weights='uniform')
# # 特征2
# knn = KNeighborsClassifier(n_neighbors=9,weights='distance')

# # test21
# # 全特征
# knn = KNeighborsClassifier(n_neighbors=3,weights='distance')
# 特征1
# knn = KNeighborsClassifier(n_neighbors=9,weights='uniform')
# 特征2
knn = KNeighborsClassifier(n_neighbors=9,weights='distance')

start_time = time.time()
knn.fit(x_train,y_train)
end_time = time.time()
train_time = end_time - start_time
print(f"knn training time: {train_time:.4f} seconds")


start_time = time.time()
y_pred = knn.predict(x_test)
end_time = time.time()
test_time = end_time - start_time
print(f"knn testing time: {test_time:.4f} seconds")
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print('Accuracy:', accuracy,"with zero_division=True")
print('Classification Report:\n', report,"with zero_division=True")

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm,"with zero_division=True")