#Realizzare programmi python per
#• Fare un plot nel tempo di ciascuno dei seguenti segnali:
#– Onda sinusoidale a 100 Hz, 200 Hz, 440 Hz
#– Onda triangolare a 100 Hz, 200 Hz, 440 Hz
#– Onda quadra a 100 Hz, 200 Hz, 440 Hz
#• Realizzare lo studio in frequenza dei segnali precedenti:
#– Trasformata di Fourier dei segnali
#– Plot di spettri di potenza, parte reale e parte immaginaria
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parametri di base
fs = 10000  # Frequenza di campionamento (10 kHz)
t = np.linspace(0, 1, fs)  # Vettore temporale da 0 a 1 secondo
frequenze = [100, 200, 440]  # Frequenze per i segnali

# Funzioni per generare i segnali
def onda_sinusoidale(f, t):
    return np.sin(2 * np.pi * f * t)

def onda_triangolare(f, t):
    return 2 * np.abs(2 * ((t * f) % 1) - 1) - 1

def onda_quadra(f, t):
    return np.sign(np.sin(2 * np.pi * f * t))

# Funzione per calcolare e plottare la trasformata di Fourier e lo spettro
def plot_fft(signal, t, f_sample):
    n = len(signal)
    # Calcolare la FFT
    fft_signal = fft(signal)
    fft_freq = fftfreq(n, 1 / f_sample)
    
    # Parte reale e immaginaria della FFT
    fft_real = np.real(fft_signal)
    fft_imag = np.imag(fft_signal)
    
    # Calcolare lo spettro di potenza
    power_spectrum = np.abs(fft_signal)**2

    # Plottare
    plt.figure(figsize=(12, 8))

    # Parte reale della FFT
    plt.subplot(3, 1, 1)
    plt.plot(fft_freq[:n//2], fft_real[:n//2])
    plt.title('Parte Reale della FFT')
    plt.xlabel('Frequenza (Hz)')
    plt.ylabel('Ampiezza')

    # Parte immaginaria della FFT
    plt.subplot(3, 1, 2)
    plt.plot(fft_freq[:n//2], fft_imag[:n//2])
    plt.title('Parte Immaginaria della FFT')
    plt.xlabel('Frequenza (Hz)')
    plt.ylabel('Ampiezza')

    # Spettro di potenza
    plt.subplot(3, 1, 3)
    plt.plot(fft_freq[:n//2], power_spectrum[:n//2])
    plt.title('Spettro di Potenza')
    plt.xlabel('Frequenza (Hz)')
    plt.ylabel('Potenza')

    plt.tight_layout()
    plt.show()

# Plottare i segnali nel dominio del tempo e FFT
for f in frequenze:
    # Generazione dei segnali
    sin_wave = onda_sinusoidale(f, t)
    tri_wave = onda_triangolare(f, t)
    sq_wave = onda_quadra(f, t)
    
    # Visualizzazione nel dominio del tempo
    plt.figure(figsize=(12, 4))
    plt.subplot(3, 1, 1)
    plt.plot(t[:1000], sin_wave[:1000])  # visualizza solo i primi 1000 campioni
    plt.title(f'Onda Sinusoidale {f} Hz')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ampiezza')
    
    plt.subplot(3, 1, 2)
    plt.plot(t[:1000], tri_wave[:1000])  # visualizza solo i primi 1000 campioni
    plt.title(f'Onda Triangolare {f} Hz')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ampiezza')

    plt.subplot(3, 1, 3)
    plt.plot(t[:1000], sq_wave[:1000])  # visualizza solo i primi 1000 campioni
    plt.title(f'Onda Quadra {f} Hz')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ampiezza')

    plt.tight_layout()
    plt.show()

    # Analisi in frequenza (FFT)
    plot_fft(sin_wave, t, fs)
    plot_fft(tri_wave, t, fs)
    plot_fft(sq_wave, t, fs)


