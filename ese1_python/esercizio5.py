import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parametri di base
fs = 10000  # Frequenza di campionamento (Hz)
t = np.linspace(0, 1, fs)  # 1 secondo di segnale

# Frequenze delle sinusoidi
frequenze = [100, 200, 440]

# Generazione delle onde sinusoidali
def onda_sinusoidale(f, t):
    return np.sin(2 * np.pi * f * t)

# Somma delle tre onde sinusoidali
segnale_somma = sum(onda_sinusoidale(f, t) for f in frequenze)

# Plot nel dominio del tempo (mostriamo solo 1000 campioni per chiarezza)
plt.figure(figsize=(10, 4))
plt.plot(t[:1000], segnale_somma[:1000])
plt.title('Segnale Somma di Sinusoidi a 100Hz, 200Hz e 440Hz')
plt.xlabel('Tempo (s)')
plt.ylabel('Ampiezza')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calcolo della FFT
n = len(segnale_somma)
fft_signal = fft(segnale_somma)
fft_freq = fftfreq(n, 1 / fs)

# Parte Reale e Immaginaria
fft_real = np.real(fft_signal)
fft_imag = np.imag(fft_signal)

# Spettro di potenza
power_spectrum = np.abs(fft_signal)**2

# Plot della FFT
plt.figure(figsize=(12, 8))

# Parte Reale
plt.subplot(3, 1, 1)
plt.plot(fft_freq[:n//2], fft_real[:n//2])
plt.title('Parte Reale della FFT (Segnale Somma)')
plt.xlabel('Frequenza (Hz)')
plt.ylabel('Ampiezza')

# Parte Immaginaria
plt.subplot(3, 1, 2)
plt.plot(fft_freq[:n//2], fft_imag[:n//2])
plt.title('Parte Immaginaria della FFT (Segnale Somma)')
plt.xlabel('Frequenza (Hz)')
plt.ylabel('Ampiezza')

# Spettro di potenza
plt.subplot(3, 1, 3)
plt.plot(fft_freq[:n//2], power_spectrum[:n//2])
plt.title('Spettro di Potenza (Segnale Somma)')
plt.xlabel('Frequenza (Hz)')
plt.ylabel('Potenza')

plt.tight_layout()
plt.show()
