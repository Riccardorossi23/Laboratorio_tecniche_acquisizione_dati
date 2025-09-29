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

# === 12. Calcola il range min-max del picco principale ===
bin_width = half_freqs[1] - half_freqs[0]
left_idx = int(results_half[2][main_peak_idx])
right_idx = int(results_half[3][main_peak_idx])
min_freq = half_freqs[left_idx]
max_freq = half_freqs[right_idx]

print(f"📊 Range di frequenze del picco principale: {min_freq:.2f} Hz – {max_freq:.2f} Hz")

# === 13. Maschera: mantieni solo il picco principale ===

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

# === 14. Trasformata inversa (IFFT) ===
filtered_signal = np.fft.ifft(filtered_fft).real

# === 15. Salva il nuovo audio ===
sf.write("solo_picco_principale.wav", filtered_signal, samplerate)
print("✅ File 'solo_picco_principale.wav' creato con solo la frequenza dominante.")
