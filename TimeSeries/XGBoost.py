import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 讀取資料
file_out_train = pd.read_csv('train.csv') 
file_out_test = pd.read_csv('test.csv')

x_train = file_out_train.iloc[:,:-1].values
y_train = file_out_train.iloc[:,-1:].astype(dtype=int).values
x_test = file_out_test.iloc[:,:-1].values
y_test = file_out_test.iloc[:,-1:].astype(dtype=int).values

# 特徵標準化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 將資料轉換為XGBoost格式
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

# 設置參數
num_classes = len(pd.unique(file_out_train.iloc[:,-1]))  # 獲取類別數量
params = {
    'objective': 'multi:softmax',  # 多類別問題，使用 softmax 激活函數
    'num_class': num_classes,  # 類別數量
    'max_depth': 6,  # 樹的最大深度
    'eta': 0.3,  # 學習率
    'eval_metric': 'merror'  # 評估指標
}

# 訓練模型
num_round = 50  # 迭代次數
bst = xgb.train(params, dtrain, num_round)

# 預測
preds = bst.predict(dtest)

# 在測試集上評估模型
accuracy = (preds == y_test).mean()
print(f'模型在測試集上的準確率：{accuracy}')


# 計算精確度、召回率、F1值（多類別）
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average='micro')
recall = recall_score(y_test, preds, average='micro')
f1 = f1_score(y_test, preds, average='micro')

print(f'準確率：{accuracy}')
print(f'精確度：{precision}')
print(f'召回率：{recall}')
print(f'F1值：{f1}')