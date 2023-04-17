# 导入所需库
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import time
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 读取数据集
data = pd.read_csv('KDDTrain+.txt', header=None)

# 设置列名
col_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes','dst_bytes', 'land',
             'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
             'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
             'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
             'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
             'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
             'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
             'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
             'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
             'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'unknown']

data.columns = col_names

# 删除无用的列
data.drop('unknown', axis=1, inplace=True)

# 标记类别为1（ddos）或0（非ddos）
data['label'] = np.where(data['label'].str.contains('neptune|smurf|back|teardrop|pod|land|apache2|udpstorm|processtable|mailbomb', case=False, regex=True), 1, 0)

# 将字符类型的特征转换为数值类型
le = LabelEncoder()
ohe = OneHotEncoder()
data.protocol_type = le.fit_transform(data.protocol_type)
data.service = le.fit_transform(data.service)
data.flag = le.fit_transform(data.flag)
data = pd.get_dummies(data)

# 特征选择
data = data.iloc[:,[0,7,9,10,15,16,19,11,22,27,28,30,41]]

# 对数据集进行标准化处理
scaler = StandardScaler()
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])


X_train = data.iloc[:, :-1]
y_train = data.iloc[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 处理训练集，并保存到 CSV 文件中
train_data = pd.DataFrame(X_train, columns=['feature_{}'.format(i+1) for i in range(len(X_train[0]))])
train_data['label'] = y_train
train_data.to_csv('train_processed.csv', index=False)

#测试集
data = pd.read_csv('KDDTest+.txt', header=None)

data.columns = col_names

# 删除无用的列
data.drop('unknown', axis=1, inplace=True)

# 标记类别为1（ddos）或0（非ddos）
data['label'] = np.where(data['label'].str.contains('neptune|smurf|back|teardrop|pod|land|apache2|udpstorm|processtable|mailbomb', case=False, regex=True), 1, 0)

# 将字符类型的特征转换为数值类型
le = LabelEncoder()
ohe = OneHotEncoder()
data.protocol_type = le.fit_transform(data.protocol_type)
data.service = le.fit_transform(data.service)
data.flag = le.fit_transform(data.flag)
data = pd.get_dummies(data)

# 特征选择
data = data.iloc[:,[0,7,9,10,15,16,19,11,22,27,28,30,41]]

# 对数据集进行标准化处理
scaler = StandardScaler()
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

X_test = data.iloc[:, :-1]
y_test = data.iloc[:, -1]

# 处理测试集，并保存到 CSV 文件中
test_data = pd.DataFrame(X_train, columns=['feature_{}'.format(i+1) for i in range(len(X_train[0]))])
test_data['label'] = y_train
test_data.to_csv('test_processed.csv', index=False)

# # Fitting SVM with the training set
# SVM = SVC(kernel='linear', random_state=0)
# start_time = time.time()
# SVM.fit(X_train, y_train)
# end_time = time.time()
# train_time = end_time - start_time
# print(f"SVM training time: {train_time:.4f} seconds")
#
#
# start_time = time.time()
# y_pred = SVM.predict(X_test)
# end_time = time.time()
# test_time = end_time - start_time
# print(f"SVM testing time: {test_time:.4f} seconds")
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)
# print('Accuracy:', accuracy,"with zero_division=True")
# print('Classification Report:\n', report,"with zero_division=True")
#
# # Evaluate the model
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", cm,"with zero_division=True")
# print()

# 朴素贝叶斯
NB=GaussianNB()
start_time = time.time()
NB.fit(X_train, y_train)
end_time = time.time()
train_time = end_time - start_time
print(f"NB training time: {train_time:.4f} seconds")

start_time = time.time()
y_pred = NB.predict(X_test)
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
print()

# knn
KNN = KNeighborsClassifier(n_neighbors=5)
start_time = time.time()
KNN.fit(X_train, y_train)
end_time = time.time()
train_time = end_time - start_time
print(f"knn training time: {train_time:.4f} seconds")

start_time = time.time()
y_pred = KNN.predict(X_test)
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
print()

# RF
RF = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
start_time = time.time()
RF.fit(X_train,y_train)
end_time = time.time()
train_time = end_time - start_time
print(f"rf training time: {train_time:.4f} seconds")

start_time = time.time()
y_pred = RF.predict(X_test)
end_time = time.time()
test_time = end_time - start_time
print(f"rf testing time: {test_time:.4f} seconds")
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print('Accuracy:', accuracy,"with zero_division=True")
print('Classification Report:\n', report,"with zero_division=True")

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm,"with zero_division=True")
print()

# DT
DT = DecisionTreeClassifier(max_depth=6, random_state=1)
start_time = time.time()
DT.fit(X_train, y_train)
end_time = time.time()
train_time = end_time - start_time
print(f"dt training time: {train_time:.4f} seconds")

start_time = time.time()
y_pred = DT.predict(X_test)
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
# print()
#
# #
# estimator_list = [
#     ('SVM',SVM),
#     ('NB',NB),
#     ('KNN',KNN),
#     ('RF',RF),
#     ('DT',DT)
#     ]
#
# # Build and fit stack model
# stack_model = StackingClassifier(
#     estimators=estimator_list, final_estimator=LogisticRegression())
# start_time = time.time()
# stack_model.fit(X_train, y_train)
# end_time = time.time()
# train_time = end_time - start_time
# print(f"model training time: {train_time:.4f} seconds")
#
# start_time = time.time()
# y_pred = stack_model.predict(X_test)
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
