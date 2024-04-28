import pywt
from matplotlib import pyplot as plt
import numpy as np
from scipy import fft
import pandas as pd

x = np.genfromtxt('3E.csv', delimiter=',', usecols=[2])

#x = x[800000:1750000]
x = x[750000:1750000]
N = len(x)
T = 1.0 / 25600.0

coeffs = pywt.wavedec(x,'db1', level=10, mode='periodic')
cA,cD10,cD9,cD8,cD7,cD6,cD5,cD4,cD3,cD2,cD1 = coeffs
#print(coeffs)
#X = range(len(x))
#print(X)

d1 = pywt.waverec(np.multiply(coeffs,[0,0,0,0,0,0,0,0,0,0,1]).tolist(),'db1', mode = 'periodic') #IDWT
d2 = pywt.waverec(np.multiply(coeffs,[0,0,0,0,0,0,0,0,0,1,0]).tolist(),'db1', mode = 'periodic')
d3 = pywt.waverec(np.multiply(coeffs,[0,0,0,0,0,0,0,0,1,0,0]).tolist(),'db1', mode = 'periodic')
d4 = pywt.waverec(np.multiply(coeffs,[0,0,0,0,0,0,0,1,0,0,0]).tolist(),'db1', mode = 'periodic')
d5 = pywt.waverec(np.multiply(coeffs,[0,0,0,0,0,0,1,0,0,0,0]).tolist(),'db1', mode = 'periodic')
d6 = pywt.waverec(np.multiply(coeffs,[0,0,0,0,0,1,0,0,0,0,0]).tolist(),'db1', mode = 'periodic')
d7 = pywt.waverec(np.multiply(coeffs,[0,0,0,0,1,0,0,0,0,0,0]).tolist(),'db1', mode = 'periodic')
d8 = pywt.waverec(np.multiply(coeffs,[0,0,0,1,0,0,0,0,0,0,0]).tolist(),'db1', mode = 'periodic')
d9 = pywt.waverec(np.multiply(coeffs,[0,0,1,0,0,0,0,0,0,0,0]).tolist(),'db1', mode = 'periodic')
d10 = pywt.waverec(np.multiply(coeffs,[0,1,0,0,0,0,0,0,0,0,0]).tolist(),'db1', mode = 'periodic')
a = pywt.waverec(np.multiply(coeffs,[1,0,0,0,0,0,0,0,0,0,0]).tolist(),'db1', mode = 'periodic')

print(pywt.dwt_max_level(N, 'db1'))


fft_result = fft.rfft(d1) 
frequencies = fft.rfftfreq(N,T)

Amplitude = np.abs(fft_result)
Amplitude_minmax = Amplitude / np.max(Amplitude)

plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.plot(d1)
plt.xlabel('Time (Samples)', fontsize = 10)
plt.ylabel('IMF1', fontsize = 10)
plt.title('IMF1', fontsize = 12)

plt.subplot(312)
plt.plot(frequencies, Amplitude / np.max(Amplitude))
plt.xlabel('Frequency (Hz)', fontsize=10)
plt.ylabel('Amplitude', fontsize=10)
plt.title('FFT Spectrum', fontsize=12)
plt.grid()

# 繪製部分頻譜圖
plt.subplot(313)
plt.plot(frequencies, Amplitude / np.max(Amplitude))
plt.xlabel('Frequency (Hz)', fontsize=10)
plt.ylabel('Amplitude', fontsize=10)
plt.title('Partial FFT Spectrum', fontsize=15)
plt.grid()
plt.show()


fft_result = fft.rfft(d3) 
frequencies = fft.rfftfreq(N,T)

Amplitude = np.abs(fft_result)
Amplitude_minmax = Amplitude / np.max(Amplitude)

plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.plot(d3)
plt.xlabel('Time (Samples)', fontsize = 10)
plt.ylabel('IMF1', fontsize = 10)
plt.title('IMF1', fontsize = 12)

plt.subplot(312)
plt.plot(frequencies, Amplitude / np.max(Amplitude))
plt.xlabel('Frequency (Hz)', fontsize=10)
plt.ylabel('Amplitude', fontsize=10)
plt.title('FFT Spectrum', fontsize=12)
plt.grid()

# 繪製部分頻譜圖
plt.subplot(313)
plt.plot(frequencies, Amplitude / np.max(Amplitude))
plt.xlabel('Frequency (Hz)', fontsize=10)
plt.ylabel('Amplitude', fontsize=10)
plt.title('Partial FFT Spectrum', fontsize=15)
plt.grid()
plt.show()

plt.figure(figsize=(16,16))
plt.subplots_adjust(hspace=1)

plt.subplot(4,1,1)
plt.plot(d1)
plt.xlabel('Time (Samples)', fontsize = 12)
plt.ylabel('D1', fontsize = 12)
plt.title("Detail Component in Level 1", fontsize = 15)

plt.subplot(4,1,2)
plt.plot(d2)
plt.xlabel('Time (Samples)', fontsize = 12)
plt.ylabel('D2', fontsize = 12)
plt.title("Detail Component in Level 2", fontsize = 15)

plt.subplot(4,1,3)
plt.plot(d3)
plt.xlabel('Time (Samples)', fontsize = 12)
plt.ylabel('D3', fontsize = 12)
plt.title("Detail Component in Level 3", fontsize = 15)

plt.subplot(4,1,4)
plt.plot(d4)
plt.xlabel('Time (Samples)', fontsize = 12)
plt.ylabel('D4', fontsize = 12)
plt.title("Detail Component in Level 4", fontsize = 15)
plt.show()


plt.figure(figsize=(16,16))
plt.subplots_adjust(hspace=1)

plt.subplot(4,1,1)
plt.plot(d5)
plt.xlabel('Time (Samples)', fontsize = 12)
plt.ylabel('D5', fontsize = 12)
plt.title("Detail Component in Level 5", fontsize = 15)

plt.subplot(4,1,2)
plt.plot(d6)
plt.xlabel('Time (Samples)', fontsize = 12)
plt.ylabel('D6', fontsize = 12)
plt.title("Detail Component in Level 6", fontsize = 15)

plt.subplot(4,1,3)
plt.plot(d7)
plt.xlabel('Time (Samples)', fontsize = 12)
plt.ylabel('D7', fontsize = 12)
plt.title("Detail Component in Level 7", fontsize = 15)

plt.subplot(4,1,4)
plt.plot(d8)
plt.xlabel('Time (Samples)', fontsize = 12)
plt.ylabel('D8', fontsize = 12)
plt.title("Detail Component in Level 8", fontsize = 15)
plt.show()



plt.figure(figsize=(16,16))
plt.subplots_adjust(hspace=1)

plt.subplot(4,1,1)
plt.plot(d9)
plt.xlabel('Time (Samples)', fontsize = 12)
plt.ylabel('D9', fontsize = 12)
plt.title("Detail Component in Level 9", fontsize = 15)

plt.subplot(4,1,2)
plt.plot(d10)
plt.xlabel('Time (Samples)', fontsize = 12)
plt.ylabel('D10', fontsize = 12)
plt.title("Detail Component in Level 10", fontsize = 15)

plt.subplot(4,1,3)
plt.plot(a)
plt.xlabel('Time (Samples)', fontsize = 12)
plt.ylabel('res.', fontsize = 12)
plt.title("Approximated Component in Level 10", fontsize = 15)
plt.show()

pd.DataFrame(d5).to_csv('3EWT.csv', index=False, header=None)
#plt.savefig('multi_level.png', dpi = 200)