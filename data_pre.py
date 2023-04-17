# 导入所需库
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

def dataprocess(source_filename,finalname):
    # 读取数据集
    data = pd.read_csv(source_filename, header=None)

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

    # # 特征选择
    # data = data.iloc[:,[0,7,9,10,15,16,19,11,22,27,28,30,41]]

    # 对数据集进行标准化处理
    scaler = StandardScaler()
    data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])
    data.to_csv(finalname, index=False)


#
# dataprocess('KDDTrain+.txt','traindata.csv')
# dataprocess('KDDTest+.txt','testdata.csv')
# dataprocess('KDDTest-21.txt','testdata-21.csv')
