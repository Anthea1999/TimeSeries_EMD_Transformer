import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input, Reshape, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

# 读取训练数据
train_data = pd.read_csv('train_data.csv')

# 读取测试数据
test_data = pd.read_csv('test_data.csv')

# 获取训练数据的时间序列和标签
X_train = train_data.iloc[:, :2500].values  # 前2500列为特征
y_train = train_data.iloc[:, 2500].values   # 最后一列为标签

# 获取测试数据的时间序列和标签
X_test = test_data.iloc[:, :2500].values
y_test = test_data.iloc[:, 2500].values

# 将标签进行编码
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_train_one_hot = to_categorical(y_train_encoded)

y_test_encoded = label_encoder.transform(y_test)
y_test_one_hot = to_categorical(y_test_encoded)

# 将时间序列数据转换为50x50的二维数据
reshaped_X_train = X_train.reshape(-1, 50, 50, 1)
reshaped_X_test = X_test.reshape(-1, 50, 50, 1)

# ResNet-18模型定义
def resnet_block(x, filters, kernel_size=3, stride=1, conv_first=True):
    if conv_first:
        x = Conv2D(filters, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    else:
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), padding='same')(x)
    return x

def resnet_18(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)

    x = resnet_block(input_tensor, filters=64, conv_first=True)
    x = resnet_block(x, filters=64)
    x = resnet_block(x, filters=128, stride=2)
    x = resnet_block(x, filters=128)
    x = resnet_block(x, filters=256, stride=2)
    x = resnet_block(x, filters=256)
    x = resnet_block(x, filters=512, stride=2)
    x = resnet_block(x, filters=512)
    x = GlobalAveragePooling2D()(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)
    return model

# 输入数据的形状
input_shape = (50, 50, 1)

# 类别数量
num_classes = len(label_encoder.classes_)

# 构建ResNet-18模型
model = resnet_18(input_shape, num_classes)

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(reshaped_X_train, y_train_one_hot, epochs=10, batch_size=32, validation_data=(reshaped_X_test, y_test_one_hot))

# 评估模型
eval_result = model.evaluate(reshaped_X_test, y_test_one_hot)

# 输出评估结果
print("Test Loss:", eval_result[0])
print("Test Accuracy:", eval_result[1])s