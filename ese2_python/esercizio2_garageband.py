import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

# === 1. Caricamento del file audio ===
filename = 'diapason.wav'  # Sostituisci con il nome corretto del file .wav
data, samplerate = sf.read(filename)

# === 2. Estrai solo un canale (es. il primo) ===
if data.ndim > 1:
    mono_data = data[:, 0]
else:
    mono_data = data  # già mono

# === 3. Plot della waveform ===
times = np.arange(len(mono_data)) / samplerate

plt.figure(figsize=(12, 4))
plt.plot(times, mono_data)
plt.title('Waveform (Primo Canale)')
plt.xlabel('Tempo [s]')
plt.ylabel('Ampiezza')
plt.grid(True)
plt.tight_layout()
plt.show()

# === 4. Scrivi un nuovo file audio identico ===
sf.write('output.wav', data, samplerate)
print("File salvato come 'output.wav'")