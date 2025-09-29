import numpy as np
import matplotlib.pyplot as plt

# Definiamo il tempo di campionamento
fs = 5000  # Frequenza di campionamento in Hz
t = np.linspace(0, 1, fs, endpoint=False)  # Un secondo di tempo

# Frequenze delle onde
frequencies = [100, 200, 440]

# Funzione per generare i segnali
def generate_signals(frequencies, t):
    signals = {}
    for f in frequencies:
        signals[f] = {
            'Onda Sinusoidale fourier': np.sin(2 * np.pi * f * t),  # Onda sinusoidale
            'Onda Triangolare fourier': 2 * np.abs(2 * (t * f - np.floor(t * f + 0.5))) - 1,  # Onda triangolare
            'Onda Quadra fourier': np.sign(np.sin(2 * np.pi * f * t))  # Onda quadra
        }
    return signals

# Funzione per calcolare la FFT
def compute_fft(signal, fs):
    fft_signal = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), 1/fs)
    return fft_freq, np.abs(fft_signal)

# Generazione dei segnali
signals = generate_signals(frequencies, t)

# Plot della FFT per ciascun segnale
fig, axs = plt.subplots(len(frequencies), 3, figsize=(15, 10))

for i, f in enumerate(frequencies):
    for j, (signal_type, signal) in enumerate(signals[f].items()):
        fft_freq, fft_signal = compute_fft(signal, fs)
        axs[i, j].plot(fft_freq[:len(fft_freq)//2], fft_signal[:len(fft_signal)//2])  # Solo la metà positiva
        axs[i, j].set_title(f'FFT {signal_type} {f} Hz')
        axs[i, j].set_xlabel('Frequenza (Hz)')
        axs[i, j].set_ylabel('Ampiezza')
        axs[i, j].grid(True)

plt.tight_layout()
plt.show()


