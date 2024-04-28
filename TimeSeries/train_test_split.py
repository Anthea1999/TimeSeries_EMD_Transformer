import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

normal_train = pd.read_csv('normal.csv')
normal_train_data, normal_test_data = train_test_split(normal_train, test_size=0.2, random_state=42)
print(len(normal_train_data))
print(len(normal_test_data))


warning_train = pd.read_csv('warning.csv')
warning_train_data, warning_test_data = train_test_split(warning_train, test_size=0.2, random_state=42)
print(len(warning_train_data))
print(len(warning_test_data))


failure_train = pd.read_csv('failure.csv')
failure_train_data, failure_test_data = train_test_split(failure_train, test_size=0.2, random_state=42)
print(len(failure_train_data))
print(len(failure_test_data))


train = np.concatenate((normal_train_data, warning_train_data, failure_train_data), axis=0)
test = np.concatenate((normal_test_data, warning_test_data, failure_test_data), axis=0)

train = pd.DataFrame(train)
test = pd.DataFrame(test)

#train.to_csv('train.csv', index=False, header=None)
#test.to_csv('test.csv', index=False, header=None)

train_random = train.sample(frac=1, random_state=1)
test_random = test.sample(frac=1, random_state=1)

train_random.to_csv('train.csv', index=False, header=None)
test_random.to_csv('test.csv', index=False, header=None)