import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# 建立並訓練Random Forest模型
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)  # 建立Random Forest分類器，設置100棵樹
random_forest.fit(x_train, y_train.ravel())  # 訓練模型

# 在驗證集上評估模型
accuracy = random_forest.score(x_val, y_val)
print(accuracy)