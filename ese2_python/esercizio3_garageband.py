import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# === 1. Leggi il file audio (.wav) ===
filename = 'distorta.wav'  # Cambia con il nome reale
data, samplerate = sf.read(filename)

# === 2. Estrai solo un canale (es. il primo) ===
if data.ndim > 1:
    mono = data[:, 0]
else:
    mono = data

# === 3. Plotta la waveform ===
time = np.arange(len(mono)) / samplerate
plt.figure(figsize=(10, 4))
plt.plot(time, mono)
plt.title('Waveform (1° canale)')
plt.xlabel('Tempo [s]')
plt.ylabel('Ampiezza')
plt.grid(True)
plt.tight_layout()
plt.show()

# === 4. Scrivi un nuovo file .wav identico ===
sf.write('output.wav', data, samplerate)
print("Nuovo file scritto come 'output.wav'.")

# === 5. Calcola la FFT ===
fft_result = np.fft.fft(mono)
fft_freq = np.fft.fftfreq(len(mono), d=1/samplerate)

# === 6. Calcola potenza, parte reale e immaginaria ===
power = np.abs(fft_result)**2
real = np.real(fft_result)
imag = np.imag(fft_result)

# === 7. Plot della potenza ===
plt.figure(figsize=(10, 4))
plt.plot(fft_freq[:len(fft_freq)//2], power[:len(power)//2])
plt.title('Spettro di Potenza (FFT)')
plt.xlabel('Frequenza [Hz]')
plt.ylabel('Potenza')
plt.grid(True)
plt.tight_layout()
plt.show()

# === 8. Parte reale e immaginaria ===
plt.figure(figsize=(10, 4))
plt.plot(fft_freq[:len(fft_freq)//2], real[:len(real)//2], label='Parte reale')
plt.plot(fft_freq[:len(fft_freq)//2], imag[:len(imag)//2], label='Parte immaginaria', linestyle='dashed')
plt.title('Parte Reale e Immaginaria della FFT')
plt.xlabel('Frequenza [Hz]')
plt.ylabel('Ampiezza')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
