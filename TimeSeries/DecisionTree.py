import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# 讀取資料
file_out_train = pd.read_csv('train.csv') 
x_train = file_out_train.iloc[:,:-1].values
y_train = file_out_train.iloc[:,-1:].astype(dtype=int).values

# 切分訓練集和驗證集
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15)

# 特徵標準化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# 建立並訓練Decision Tree模型
decision_tree = DecisionTreeClassifier()  # 建立Decision Tree分類器
decision_tree.fit(x_train, y_train.ravel())  # 訓練模型

# 在驗證集上評估模型
accuracy = decision_tree.score(x_val, y_val)
print(accuracy)
