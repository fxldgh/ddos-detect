# 加载模型
import time
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 读取数据集
test_data = pd.read_csv('testdata.csv', header=None)
print(test_data.shape)

x_test = test_data.iloc[1:,:-1].astype('float')
y_test = test_data.iloc[1:,-1].astype('float')


test21_data = pd.read_csv('testdata-21.csv', header=None)
print(test_data.shape)

x_test21 = test21_data.iloc[1:,:-1].astype('float')
y_test21 = test21_data.iloc[1:,-1].astype('float')


clf = load('model_test.dat')

# 使用模型进行预测
start_time = time.time()
y_pred = clf.predict(x_test)
end_time = time.time()
test_time = end_time - start_time
print(f"model testing time: {test_time:.4f} seconds")
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("stacked model")
print('Accuracy:', accuracy,"with zero_division=True")
print('Classification Report:\n', report,"with zero_division=True")

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm,"with zero_division=True")
print()



clf_21 = load('model_test21.dat')

# 使用模型进行预测
start_time = time.time()
y_pred21 = clf_21.predict(x_test21)
end_time = time.time()
test_time = end_time - start_time
print(f"model testing time: {test_time:.4f} seconds")
accuracy = accuracy_score(y_test21, y_pred21)
report = classification_report(y_test21, y_pred21)
print("stacked model")
print('Accuracy:', accuracy,"with zero_division=True")
print('Classification Report:\n', report,"with zero_division=True")

# Evaluate the model
cm = confusion_matrix(y_test21, y_pred21)
print("Confusion Matrix:\n", cm,"with zero_division=True")




