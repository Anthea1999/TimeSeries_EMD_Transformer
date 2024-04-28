import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 讀取訓練資料
file_out_train = pd.read_csv('test.csv') 
x_train = file_out_train.iloc[:,:-1].values
y_train = file_out_train.iloc[:,-1:].astype(dtype=int).values

# 切分訓練集和驗證集
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15)

# 特徵標準化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# 建立並訓練SVM模型
svm = SVC(kernel='rbf', C=1.0, gamma='scale')  # 使用RBF核心的SVM，C為正則化參數，gamma為RBF核參數
svm.fit(x_train, y_train.ravel())  # 訓練模型

# 在驗證集上評估模型
accuracy = svm.score(x_val, y_val)
print(f'模型在驗證集上的準確率：{accuracy}')

# 讀取測試資料
file_out_test = pd.read_csv('test.csv') 
x_test = file_out_test.iloc[:,:-1].values
y_test = file_out_test.iloc[:,-1:].astype(dtype=int).values

# 對測試資料進行特徵標準化
x_test = scaler.transform(x_test)  # 使用相同的標準化參數

# 預測測試資料
y_pred = svm.predict(x_test)

# 評估預測結果
test_accuracy = svm.score(x_test, y_test)
print(f'模型在測試集上的準確率：{test_accuracy}')


conf_matrix = confusion_matrix(y_test, y_pred)
print(f'混淆矩陣：\n{conf_matrix}')

# 計算准確率、精確度、召回率、F1值
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')

print(f'準確率：{accuracy}')
print(f'精確度：{precision}')
print(f'召回率：{recall}')
print(f'F1值：{f1}')