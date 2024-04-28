import pandas as pd
import numpy as np

normal_data = pd.read_csv('normal_data.csv')
warning_data = pd.read_csv('warning_data.csv')
failure_data = pd.read_csv('failure_data.csv')

def sliding_window(signal, window_size, step):
    result = []
    for i in range(0, len(signal) - window_size + 1, step):
        window = signal[i:i + window_size]
        result.append(window)
    result = np.reshape(result,(len(result),window_size))
    return result

window_size = 2500  # 指定窗口大小
step = 10  # 指定步長

normal = sliding_window(normal_data, window_size, step)
normal = np.insert(normal, normal.shape[1], 0, axis=1)
pd.DataFrame(normal).to_csv('normal.csv', index=False, header=None)
print(len(normal))

warning = sliding_window(warning_data, window_size, step)
warning = np.insert(warning, warning.shape[1], 1, axis=1)
pd.DataFrame(warning).to_csv('warning.csv', index=False, header=None)
print(len(warning))

failure = sliding_window(failure_data, window_size, step)
failure = np.insert(failure, failure.shape[1], 2, axis=1)
pd.DataFrame(failure).to_csv('failure.csv', index=False, header=None)
print(len(failure))