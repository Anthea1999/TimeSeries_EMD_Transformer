import pandas as pd
import numpy as np
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

# 模擬機器行為波形數據，這里假設data是一個包含行為數據的列表
#data = np.genfromtxt('imf6.csv', delimiter=',', usecols=[0])
data = np.genfromtxt('1_preprocess.csv', delimiter=',', usecols=[0])
#data = data[1500000:]
#pd.DataFrame(data).to_csv('imf6_preprocess.csv', index=False, header=None)
plt.figure(figsize=(16, 8))
plt.plot(data, 'k')
plt.show()

fs = 32768

# 计算时频图
frequencies, times, Sxx = spectrogram(data, fs)

# 绘制时频图
plt.figure(figsize=(16, 8))
cm=plt.cm.get_cmap('rainbow')
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), cmap=cm, alpha=1 , shading='auto')  # 以对数刻度绘制功率谱密度
plt.title('Spectrogram of the Signal')
plt.xlabel('Time (min)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.show()

#拆分
window_size = 2250
step_size = 325
threshold_fraction = 0.5  # 標準差的一部分
previous_window_data = None

# 初始化狀態列表
index = []
class_index = []

for i in range(0, len(data) - window_size + 1, step_size):
    window_data = data[i:i+window_size]

    if previous_window_data is not None:
        current_window_average = np.mean(window_data)
        previous_window_average = np.mean(previous_window_data)
        
        previous_window_stddev = np.std(previous_window_data)

        # 计算第一个窗口标准差的一部分
        threshold = threshold_fraction * previous_window_stddev
        
        if abs(current_window_average - previous_window_average) > threshold:
            # 处理超出阈值的情况，例如记录或采取适当的操作
            index.append(i)

    previous_window_data = window_data


print(index)

for index_value in range(len(index)-1):
    if index[index_value+1] - index[index_value] > 10000:
        class_index.append(index[index_value+1])

print(class_index)

warning = class_index[13]
failure = class_index[-1]

print(warning)
print(failure)


plt.figure(figsize=(16, 8))
# 假設要在索引處更改顏色
# 將索引之前的部分繪製成藍色
plt.plot(range(warning), data[:warning], color='blue')
# 將索引之後的部分繪製成紅色
plt.plot(range(warning, failure), data[warning:failure], color='orange')
plt.plot(range(failure, len(data)), data[failure:], color='red')
plt.show()

normal_data = data[:warning]
warning_data = data[warning:failure]
failure_data = data[failure:]

pd.DataFrame(normal_data).to_csv('normal_data.csv', index=False, header=None) 
pd.DataFrame(warning_data).to_csv('warning_data.csv', index=False, header=None) 
pd.DataFrame(failure_data).to_csv('failure_data.csv', index=False, header=None) 

# 计算时频图
frequencies_nor, times_nor, Sxx_nor = spectrogram(normal_data, fs)
frequencies_warning, times_warning, Sxx_warning = spectrogram(warning_data, fs)
frequencies_failure, times_failure, Sxx_failure = spectrogram(failure_data, fs)

# 绘制时频图
plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)

cm=plt.cm.get_cmap('rainbow')

plt.subplot(311)
plt.pcolormesh(times_nor, frequencies_nor, 10 * np.log10(Sxx_nor), cmap=cm, alpha=1 , shading='auto')  # 以对数刻度绘制功率谱密度
plt.title('Spectrogram of the Signal')
plt.xlabel('Time (min)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power/Frequency (dB/Hz)')

plt.subplot(312)
plt.pcolormesh(times_warning, frequencies_warning, 10 * np.log10(Sxx_warning), cmap=cm, alpha=1 , shading='auto')  # 以对数刻度绘制功率谱密度
plt.title('Spectrogram of the Signal')
plt.xlabel('Time (min)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power/Frequency (dB/Hz)')

plt.subplot(313)
plt.pcolormesh(times_failure, frequencies_failure, 10 * np.log10(Sxx_failure), cmap=cm, alpha=1 , shading='auto')  # 以对数刻度绘制功率谱密度
plt.title('Spectrogram of the Signal')
plt.xlabel('Time (min)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.show()