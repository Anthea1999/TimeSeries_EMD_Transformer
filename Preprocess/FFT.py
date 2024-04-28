import emd
import math
import pandas as pd
import numpy as np
from scipy import signal
from scipy import fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pywt
import time

entropys = []
peak_frequencies = []
Amplitudes = []

window_size = 2360
step_size = 2360

#x = np.genfromtxt('all.csv', delimiter=',', usecols=[0])
#x = x[4147000:7147000]

x = np.genfromtxt('all1-1.csv', delimiter=',', usecols=[0])
x = x[2030464:]   #資料裁切

pd.DataFrame(x).to_csv('all_preprocess.csv', index=False, header=None)
plt.figure(figsize=(16, 8))
plt.plot(x)
plt.show()



imf = emd.sift.sift(x)
print(imf.shape)

def my_get_next_imf(x, zoom=None, sd_thresh=0.1):

    proto_imf = x.copy()  # Take a copy of the input so we don't overwrite anything
    continue_sift = True  # Define a flag indicating whether we should continue sifting
    niters = 0            # An iteration counter

    if zoom is None:
        zoom = (0, x.shape[0])

    # Main loop - we don't know how many iterations we'll need so we use a ``while`` loop
    while continue_sift:
        niters += 1  # Increment the counter

        # Compute upper and lower envelopes
        upper_env = emd.sift.interp_envelope(proto_imf, mode='upper')
        lower_env = emd.sift.interp_envelope(proto_imf, mode='lower')

        # Compute average envelope
        avg_env = (upper_env+lower_env) / 2

        # Add a summary subplot
        #plt.figure(figsize=(12, 10))
        #plt.subplot(5, 1, niters)
        #plt.plot(proto_imf[zoom[0]:zoom[1]], 'k')
        #plt.plot(upper_env[zoom[0]:zoom[1]])
        #plt.plot(lower_env[zoom[0]:zoom[1]])
        #plt.plot(avg_env[zoom[0]:zoom[1]])

        # Should we stop sifting?
        stop, val = emd.sift.sd_stop(proto_imf-avg_env, proto_imf, sd=sd_thresh)

        # Remove envelope from proto IMF
        proto_imf = proto_imf - avg_env

        # and finally, stop if we're stopping
        if stop:
            continue_sift = False

    # Return extracted IMF
    return proto_imf

def calculate_entropy(probabilities):
    entropy = 0
    for probability in probabilities:
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return entropy

start_time = time.time()

imf1 = my_get_next_imf(x)
imf2 = my_get_next_imf(x - imf1)
imf3 = my_get_next_imf(x - imf1 - imf2)
imf4 = my_get_next_imf(x - imf1 - imf2 - imf3)
imf5 = my_get_next_imf(x - imf1 - imf2 - imf3 - imf4)
imf6 = my_get_next_imf(x - imf1 - imf2 - imf3 - imf4 - imf5)
imf7 = my_get_next_imf(x - imf1 - imf2 - imf3 - imf4 - imf5 - imf6)
imf8 = my_get_next_imf(x - imf1 - imf2 - imf3 - imf4 - imf5 - imf6 - imf7)
imf9 = my_get_next_imf(x - imf1 - imf2 - imf3 - imf4 - imf5 - imf6 - imf7 - imf8)
imf10 = my_get_next_imf(x - imf1 - imf2 - imf3 - imf4 - imf5 - imf6 - imf7 - imf8 - imf9)
imf11 = my_get_next_imf(x - imf1 - imf2 - imf3 - imf4 - imf5 - imf6 - imf7 - imf8 - imf9 - imf10)
imf12 = my_get_next_imf(x - imf1 - imf2 - imf3 - imf4 - imf5 - imf6 - imf7 - imf8 - imf9 - imf10 - imf11)
imf13 = my_get_next_imf(x - imf1 - imf2 - imf3 - imf4 - imf5 - imf6 - imf7 - imf8 - imf9 - imf10 - imf11 - imf12)
imf14 = my_get_next_imf(x - imf1 - imf2 - imf3 - imf4 - imf5 - imf6 - imf7 - imf8 - imf9 - imf10 - imf11 - imf12 - imf13)
imf15 = x - imf1 - imf2 - imf3 - imf4 - imf5 - imf6 - imf7 - imf8 - imf9 - imf10 - imf11 - imf12 - imf13 - imf14

N = len(x)
T = 1.0 / 25600.0


fft_result = fft.rfft(imf1) 
frequencies = fft.rfftfreq(N,T)

Amplitude = np.abs(fft_result)
Amplitude_minmax = Amplitude / np.max(Amplitude)
peaks, _ = find_peaks(Amplitude_minmax, height=0.75)
peak_frequency = frequencies[peaks]
peak_frequencies.append(peak_frequency[0])
print(peak_frequency)

for i in range(0, len(frequencies) - window_size + 1, step_size):   
    Amplitude_sum = np.sum(Amplitude[i:i+window_size])
    Amplitudes.append(Amplitude_sum)

normalized_Amplitude = Amplitudes / np.sum(Amplitudes)
entropy = calculate_entropy(normalized_Amplitude)
print(entropy)
entropys.append(entropy)

plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.plot(imf1)
plt.xlabel('Time (Samples)', fontsize = 10)
plt.ylabel('IMF1', fontsize = 10)
plt.title('IMF1', fontsize = 12)
plt.grid()

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

fft_result = fft.rfft(imf2) 
frequencies = fft.rfftfreq(N,T)

Amplitude = np.abs(fft_result)
Amplitude_minmax = Amplitude / np.max(Amplitude)
peaks, _ = find_peaks(Amplitude_minmax, height=0.75)
peak_frequency = frequencies[peaks]
peak_frequency = np.array(peak_frequency)
peak_frequencies.append(peak_frequency[0])
print(peak_frequency)

for i in range(0, len(frequencies) - window_size + 1, step_size):   
    Amplitude_sum = np.sum(Amplitude[i:i+window_size])
    Amplitudes.append(Amplitude_sum)

normalized_Amplitude = Amplitudes / np.sum(Amplitudes)
Amplitudes.clear()
entropy = calculate_entropy(normalized_Amplitude)
entropys.append(entropy)

plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.plot(imf2)
plt.xlabel('Time (Samples)', fontsize = 10)
plt.ylabel('IMF2', fontsize = 10)
plt.title('IMF2', fontsize = 12)
plt.grid()

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

fft_result = fft.rfft(imf3) 
frequencies = fft.rfftfreq(N,T)

Amplitude = np.abs(fft_result)
Amplitude_minmax = Amplitude / np.max(Amplitude)
peaks, _ = find_peaks(Amplitude_minmax, height=0.75)
peak_frequency = frequencies[peaks]
peak_frequency = np.array(peak_frequency)
peak_frequencies.append(peak_frequency[0])
print(peak_frequency)

for i in range(0, len(frequencies) - window_size + 1, step_size):   
    Amplitude_sum = np.sum(Amplitude[i:i+window_size])
    Amplitudes.append(Amplitude_sum)

normalized_Amplitude = Amplitudes / np.sum(Amplitudes)
Amplitudes.clear()
entropy = calculate_entropy(normalized_Amplitude)
entropys.append(entropy)

plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.plot(imf3)
plt.xlabel('Time (Samples)', fontsize = 10)
plt.ylabel('IMF3', fontsize = 10)
plt.title('IMF3', fontsize = 12)
plt.grid()

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

fft_result = fft.rfft(imf4) 
frequencies = fft.rfftfreq(N,T)

Amplitude = np.abs(fft_result)
Amplitude_minmax = Amplitude / np.max(Amplitude)
peaks, _ = find_peaks(Amplitude_minmax, height=0.75)
peak_frequency = frequencies[peaks]
peak_frequency = np.array(peak_frequency)
peak_frequencies.append(peak_frequency[0])
print(peak_frequency)

for i in range(0, len(frequencies) - window_size + 1, step_size):   
    Amplitude_sum = np.sum(Amplitude[i:i+window_size])
    Amplitudes.append(Amplitude_sum)

normalized_Amplitude = Amplitudes / np.sum(Amplitudes)
Amplitudes.clear()
entropy = calculate_entropy(normalized_Amplitude)
entropys.append(entropy)

plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.plot(imf4)
plt.xlabel('Time (Samples)', fontsize = 10)
plt.ylabel('IMF4', fontsize = 10)
plt.title('IMF4', fontsize = 12)
plt.grid()

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

fft_result = fft.rfft(imf5) 
frequencies = fft.rfftfreq(N,T)

Amplitude = np.abs(fft_result)
Amplitude_minmax = Amplitude / np.max(Amplitude)
peaks, _ = find_peaks(Amplitude_minmax, height=0.75)
peak_frequency = frequencies[peaks]
peak_frequency = np.array(peak_frequency)
peak_frequencies.append(peak_frequency[0])
print(peak_frequency)

for i in range(0, len(frequencies) - window_size + 1, step_size):   
    Amplitude_sum = np.sum(Amplitude[i:i+window_size])
    Amplitudes.append(Amplitude_sum)

normalized_Amplitude = Amplitudes / np.sum(Amplitudes)
Amplitudes.clear()
entropy = calculate_entropy(normalized_Amplitude)
entropys.append(entropy)

plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.plot(imf5)
plt.xlabel('Time (Samples)', fontsize = 10)
plt.ylabel('IMF5', fontsize = 10)
plt.title('IMF5', fontsize = 12)
plt.grid()

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

fft_result = fft.rfft(imf6) 
frequencies = fft.rfftfreq(N,T)

Amplitude = np.abs(fft_result)
Amplitude_minmax = Amplitude / np.max(Amplitude)
peaks, _ = find_peaks(Amplitude_minmax, height=0.7)
peak_frequency = frequencies[peaks]
peak_frequency = np.array(peak_frequency)
peak_frequencies.append(peak_frequency[0])
print(peak_frequency)

for i in range(0, len(frequencies) - window_size + 1, step_size):   
    Amplitude_sum = np.sum(Amplitude[i:i+window_size])
    Amplitudes.append(Amplitude_sum)

normalized_Amplitude = Amplitudes / np.sum(Amplitudes)
Amplitudes.clear()
entropy = calculate_entropy(normalized_Amplitude)
entropys.append(entropy)

plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.plot(imf6)
plt.xlabel('Time (Samples)', fontsize = 10)
plt.ylabel('IMF6', fontsize = 10)
plt.title('IMF6', fontsize = 12)
plt.grid()

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

fft_result = fft.rfft(imf7) 
frequencies = fft.rfftfreq(N,T)

Amplitude = np.abs(fft_result)
Amplitude_minmax = Amplitude / np.max(Amplitude)
peaks, _ = find_peaks(Amplitude_minmax, height=0.75)
peak_frequency = frequencies[peaks]
peak_frequency = np.array(peak_frequency)
peak_frequencies.append(peak_frequency[0])
print(peak_frequency)

for i in range(0, len(frequencies) - window_size + 1, step_size):   
    Amplitude_sum = np.sum(Amplitude[i:i+window_size])
    Amplitudes.append(Amplitude_sum)

normalized_Amplitude = Amplitudes / np.sum(Amplitudes)
Amplitudes.clear()
entropy = calculate_entropy(normalized_Amplitude)
entropys.append(entropy)

plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.plot(imf7)
plt.xlabel('Time (Samples)', fontsize = 10)
plt.ylabel('IMF7', fontsize = 10)
plt.title('IMF7', fontsize = 12)
plt.grid()

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

fft_result = fft.rfft(imf8) 
frequencies = fft.rfftfreq(N,T)

Amplitude = np.abs(fft_result)
Amplitude_minmax = Amplitude / np.max(Amplitude)
peaks, _ = find_peaks(Amplitude_minmax, height=0.75)
peak_frequency = frequencies[peaks]
peak_frequency = np.array(peak_frequency)
peak_frequencies.append(peak_frequency[0])
print(peak_frequency)

for i in range(0, len(frequencies) - window_size + 1, step_size):   
    Amplitude_sum = np.sum(Amplitude[i:i+window_size])
    Amplitudes.append(Amplitude_sum)

normalized_Amplitude = Amplitudes / np.sum(Amplitudes)
Amplitudes.clear()
entropy = calculate_entropy(normalized_Amplitude)
entropys.append(entropy)

plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)

plt.subplot(311)
plt.plot(imf8)
plt.xlabel('Time (Samples)', fontsize = 10)
plt.ylabel('IMF8', fontsize = 10)
plt.title('IMF8', fontsize = 12)
plt.grid()

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

fft_result = fft.rfft(imf9) 
frequencies = fft.rfftfreq(N,T)

Amplitude = np.abs(fft_result)
Amplitude_minmax = Amplitude / np.max(Amplitude)
peaks, _ = find_peaks(Amplitude_minmax, height=0.75)
peak_frequency = frequencies[peaks]
peak_frequency = np.array(peak_frequency)
peak_frequencies.append(peak_frequency[0])
print(peak_frequency)

for i in range(0, len(frequencies) - window_size + 1, step_size):   
    Amplitude_sum = np.sum(Amplitude[i:i+window_size])
    Amplitudes.append(Amplitude_sum)

normalized_Amplitude = Amplitudes / np.sum(Amplitudes)
Amplitudes.clear()
entropy = calculate_entropy(normalized_Amplitude)
entropys.append(entropy)

fft_result = fft.rfft(imf10) 
frequencies = fft.rfftfreq(N,T)

Amplitude = np.abs(fft_result)
Amplitude_minmax = Amplitude / np.max(Amplitude)
peaks, _ = find_peaks(Amplitude_minmax, height=0.75)
peak_frequency = frequencies[peaks]
peak_frequency = np.array(peak_frequency)
peak_frequencies.append(peak_frequency[0])
print(peak_frequency)

for i in range(0, len(frequencies) - window_size + 1, step_size):   
    Amplitude_sum = np.sum(Amplitude[i:i+window_size])
    Amplitudes.append(Amplitude_sum)

normalized_Amplitude = Amplitudes / np.sum(Amplitudes)
Amplitudes.clear()
entropy = calculate_entropy(normalized_Amplitude)
entropys.append(entropy)

fft_result = fft.rfft(imf11) 
frequencies = fft.rfftfreq(N,T)

Amplitude = np.abs(fft_result)
Amplitude_minmax = Amplitude / np.max(Amplitude)
peaks, _ = find_peaks(Amplitude_minmax, height=0.75)
peak_frequency = frequencies[peaks]
peak_frequency = np.array(peak_frequency)
peak_frequencies.append(peak_frequency[0])
print(peak_frequency)

for i in range(0, len(frequencies) - window_size + 1, step_size):   
    Amplitude_sum = np.sum(Amplitude[i:i+window_size])
    Amplitudes.append(Amplitude_sum)

normalized_Amplitude = Amplitudes / np.sum(Amplitudes)
Amplitudes.clear()
entropy = calculate_entropy(normalized_Amplitude)
entropys.append(entropy)

fft_result = fft.rfft(imf12) 
frequencies = fft.rfftfreq(N,T)

Amplitude = np.abs(fft_result)
Amplitude_minmax = Amplitude / np.max(Amplitude)
peaks, _ = find_peaks(Amplitude_minmax, height=0.75)
peak_frequency = frequencies[peaks]
peak_frequency = np.array(peak_frequency)
peak_frequencies.append(peak_frequency[0])
print(peak_frequency)

for i in range(0, len(frequencies) - window_size + 1, step_size):   
    Amplitude_sum = np.sum(Amplitude[i:i+window_size])
    Amplitudes.append(Amplitude_sum)

normalized_Amplitude = Amplitudes / np.sum(Amplitudes)
Amplitudes.clear()
entropy = calculate_entropy(normalized_Amplitude)
entropys.append(entropy)

fft_result = fft.rfft(imf13) 
frequencies = fft.rfftfreq(N,T)

Amplitude = np.abs(fft_result)
Amplitude_minmax = Amplitude / np.max(Amplitude)
peaks, _ = find_peaks(Amplitude_minmax, height=0.75)
peak_frequency = frequencies[peaks]
peak_frequency = np.array(peak_frequency)
peak_frequencies.append(peak_frequency[0])
print(peak_frequency)

for i in range(0, len(frequencies) - window_size + 1, step_size):   
    Amplitude_sum = np.sum(Amplitude[i:i+window_size])
    Amplitudes.append(Amplitude_sum)

normalized_Amplitude = Amplitudes / np.sum(Amplitudes)
Amplitudes.clear()
entropy = calculate_entropy(normalized_Amplitude)
entropys.append(entropy)

fft_result = fft.rfft(imf14) 
frequencies = fft.rfftfreq(N,T)

Amplitude = np.abs(fft_result)
Amplitude_minmax = Amplitude / np.max(Amplitude)
peaks, _ = find_peaks(Amplitude_minmax, height=0.75)
peak_frequency = frequencies[peaks]
peak_frequency = np.array(peak_frequency)
peak_frequencies.append(peak_frequency[0])
print(peak_frequency)

for i in range(0, len(frequencies) - window_size + 1, step_size):   
    Amplitude_sum = np.sum(Amplitude[i:i+window_size])
    Amplitudes.append(Amplitude_sum)

normalized_Amplitude = Amplitudes / np.sum(Amplitudes)
Amplitudes.clear()
entropy = calculate_entropy(normalized_Amplitude)
entropys.append(entropy)
"""
fft_result = fft.rfft(imf15) 
frequencies = fft.rfftfreq(N,T)

Amplitude = np.abs(fft_result)
Amplitude_minmax = Amplitude / np.max(Amplitude)
peaks, _ = find_peaks(Amplitude_minmax, height=0.75)
peak_frequency = frequencies[peaks]
peak_frequency = np.array(peak_frequency)
peak_frequencies.append(peak_frequency[0])
print(peak_frequency)

for i in range(0, len(frequencies) - window_size + 1, step_size):   
    Amplitude_sum = np.sum(Amplitude[i:i+window_size])
    Amplitudes.append(Amplitude_sum)

normalized_Amplitude = Amplitudes / np.sum(Amplitudes)
Amplitudes.clear()
entropy = calculate_entropy(normalized_Amplitude)
entropys.append(entropy)
"""

print('entropys:', entropys)
print('peak_frequencies:', peak_frequencies)

df = pd.DataFrame({'Entropy': entropys, 'Peak frequency': peak_frequencies})
#df = pd.DataFrame({'Value': data, 'labels': labels, 'Status': states})

# 將結果保存為CSV文件
#df.to_csv('EMD selection.csv', index=False)
df.to_csv('EMD selection 1-3.csv')


data = imf8

#pd.DataFrame(imf5).to_csv('imf5_preprocess.csv', index=False, header=None) 
pd.DataFrame(data).to_csv('1_preprocess.csv', index=False, header=None) 
#pd.DataFrame(imf7).to_csv('imf7_preprocess.csv', index=False, header=None) 
