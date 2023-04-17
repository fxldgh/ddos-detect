# 导入所需库
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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



# RF
# test
# # 特征1
# RF = RandomForestClassifier(n_estimators = 93, max_features = 5, min_samples_leaf = 8, min_samples_split = 21, criterion = 'gini', random_state = 0)
# 特征2
# RF = RandomForestClassifier(n_estimators = 13, max_features = 5, min_samples_leaf = 7, min_samples_split = 21, criterion = 'entropy', random_state = 0)

# test21
# 特征1
# RF = RandomForestClassifier(n_estimators = 133, max_depth= 14, max_features = 5, min_samples_leaf = 8, min_samples_split = 21, criterion = 'gini', random_state = 0)
# 特征2
RF = RandomForestClassifier(n_estimators = 14, max_depth= 13, max_features = 5, min_samples_leaf = 7, min_samples_split = 21, criterion = 'entropy', random_state = 0)

start_time = time.time()
RF.fit(x_train, y_train)
end_time = time.time()
train_time = end_time - start_time
print(f"RF training time: {train_time:.4f} seconds")

start_time = time.time()
y_pred = RF.predict(x_test)
end_time = time.time()
test_time = end_time - start_time
print(f"RF testing time: {test_time:.4f} seconds")
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print('Accuracy:', accuracy,"with zero_division=True")
print('Classification Report:\n', report,"with zero_division=True")

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm,"with zero_division=True")