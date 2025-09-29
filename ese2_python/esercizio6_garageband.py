import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# === 1. Leggi il file audio (.wav) ===
filename = 'pulita_semplice.wav'  # Cambia con il nome reale
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
# === 9. Trova i picchi principali nello spettro di potenza ===
from scipy.signal import find_peaks

# Considera solo la metà positiva dello spettro (simmetria)
half_freqs = fft_freq[:len(fft_freq)//2]
half_power = power[:len(power)//2]

# Trova i picchi sopra una certa soglia
peaks, _ = find_peaks(half_power, height=np.max(half_power)*0.1, distance=20)

# Frequenze dei picchi
peak_freqs = half_freqs[peaks]
peak_powers = half_power[peaks]

# Picco principale
main_peak_idx = np.argmax(peak_powers)
main_freq = peak_freqs[main_peak_idx]

print(f"🎯 Frequenza principale: {main_freq:.2f} Hz")

# (Facoltativo) Picco secondario
if len(peak_freqs) > 1:
    second_idx = np.argsort(peak_powers)[-2]
    second_freq = peak_freqs[second_idx]
    print(f"🎵 Picco secondario: {second_freq:.2f} Hz")
else:
    second_freq = None

# === 10. Converti frequenza in nota musicale ===
def freq_to_note(freq):
    A4 = 440.0
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if freq == 0:
        return "N/A"
    n = round(12 * np.log2(freq / A4))
    note_index = (n + 9) % 12  # A4 = index 9
    octave = 4 + (n + 9) // 12
    return f"{note_names[note_index]}{octave}"

main_note = freq_to_note(main_freq)
print(f"📝 Nota corrispondente al picco principale: {main_note}")

if second_freq:
    second_note = freq_to_note(second_freq)
    print(f"📝 Nota secondaria (facoltativa): {second_note}")

# === 11. Larghezza del picco (approssimata come larghezza a mezza altezza) ===
from scipy.signal import peak_widths

results_half = peak_widths(half_power, peaks, rel_height=0.5)
main_width_hz = results_half[0][main_peak_idx] * (half_freqs[1] - half_freqs[0])
print(f"📏 Larghezza del picco principale: {main_width_hz:.2f} Hz")

# === 12. Maschera: mantieni solo il picco principale ===

# Converti frequenza in indice nella FFT
freq_resolution = samplerate / len(fft_result)
main_idx = int(np.round(main_freq / freq_resolution))

# Crea una maschera: finestra attorno al picco (es. ±5 bin)
width_bins = 5
mask = np.zeros_like(fft_result, dtype=bool)
mask[main_idx - width_bins : main_idx + width_bins + 1] = True
# Per simmetria (per garantire segnale reale) considera anche la parte negativa
mask[-main_idx - width_bins : -main_idx + width_bins + 1] = True

# Applica la maschera
filtered_fft = np.zeros_like(fft_result, dtype=complex)
filtered_fft[mask] = fft_result[mask]

# === 13. Trasformata inversa (IFFT) ===
filtered_signal = np.fft.ifft(filtered_fft).real

# === 14. Salva il nuovo audio ===
sf.write("solo_picco_principale.wav", filtered_signal, samplerate)
print("✅ File 'solo_picco_principale.wav' creato con solo la frequenza dominante.")

# === 15. Applica un filtro in frequenza e visualizza ===

# Parametri del filtro
filter_type = "highpass"     # "lowpass", "highpass" o "bandpass"
cutoff_low = 500             # Hz
cutoff_high = 3000           # solo per bandpass

# Calcola FFT e frequenze
fft_data = np.fft.fft(mono)
freqs = np.fft.fftfreq(len(fft_data), d=1/samplerate)
power_original = np.abs(fft_data)**2

# Crea maschera
mask = np.zeros_like(freqs, dtype=bool)

if filter_type == "lowpass":
    mask[np.abs(freqs) <= cutoff_low] = True
elif filter_type == "highpass":
    mask[np.abs(freqs) >= cutoff_low] = True
elif filter_type == "bandpass":
    mask[(np.abs(freqs) >= cutoff_low) & (np.abs(freqs) <= cutoff_high)] = True

# Applica maschera
filtered_fft = fft_data * mask
power_filtered = np.abs(filtered_fft)**2

# Inversa per ottenere segnale nel tempo
filtered_signal = np.fft.ifft(filtered_fft).real

# === 16. Salva audio filtrato ===
sf.write(f"{filter_type}_filtrato.wav", filtered_signal, samplerate)
print(f"✅ File '{filter_type}_filtrato.wav' salvato.")

# === 17. Plot ===

# Spettro originale
plt.figure(figsize=(12, 4))
plt.plot(freqs[:len(freqs)//2], power_original[:len(freqs)//2])
plt.title('Spettro Originale')
plt.xlabel('Frequenza [Hz]')
plt.ylabel('Potenza')
plt.grid(True)
plt.tight_layout()
plt.show()

# Spettro filtrato
plt.figure(figsize=(12, 4))
plt.plot(freqs[:len(freqs)//2], power_filtered[:len(freqs)//2], color='orange')
plt.title(f'Spettro dopo filtro {filter_type}')
plt.xlabel('Frequenza [Hz]')
plt.ylabel('Potenza')
plt.grid(True)
plt.tight_layout()
plt.show()

# Waveform filtrata
time = np.arange(len(filtered_signal)) / samplerate
plt.figure(figsize=(12, 4))
plt.plot(time, filtered_signal, color='green')
plt.title(f'Segno nel tempo dopo filtro {filter_type}')
plt.xlabel('Tempo [s]')
plt.ylabel('Ampiezza')
plt.grid(True)
plt.tight_layout()
plt.show()


