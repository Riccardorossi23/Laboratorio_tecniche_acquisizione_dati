import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths


def read_audio(filename):
    """Legge un file audio e lo converte in mono se necessario."""
    data, samplerate = sf.read(filename)
    # Estrai solo il primo canale se il file è stereo
    if data.ndim > 1:
        mono = data[:, 0]
    else:
        mono = data
    return data, mono, samplerate


def plot_waveform(audio, samplerate, title='Waveform (1° canale)', color=None):
    """Visualizza la forma d'onda del segnale audio."""
    time = np.arange(len(audio)) / samplerate
    plt.figure(figsize=(10, 4))
    if color:
        plt.plot(time, audio, color=color)
    else:
        plt.plot(time, audio)
    plt.title(title)
    plt.xlabel('Tempo [s]')
    plt.ylabel('Ampiezza')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_fft(audio, samplerate):
    """Calcola la FFT e le relative componenti."""
    fft_result = np.fft.fft(audio)
    fft_freq = np.fft.fftfreq(len(audio), d=1/samplerate)
    
    # Calcola potenza, parte reale e immaginaria
    power = np.abs(fft_result)**2
    real = np.real(fft_result)
    imag = np.imag(fft_result)
    
    # Solo metà dello spettro (frequenze positive)
    half_freqs = fft_freq[:len(fft_freq)//2]
    half_power = power[:len(power)//2]
    half_real = real[:len(real)//2]
    half_imag = imag[:len(imag)//2]
    
    return fft_result, fft_freq, power, half_freqs, half_power, half_real, half_imag


def plot_power_spectrum(freqs, power, title='Spettro di Potenza (FFT)', color=None):
    """Visualizza lo spettro di potenza."""
    plt.figure(figsize=(10, 4))
    if color:
        plt.plot(freqs, power, color=color)
    else:
        plt.plot(freqs, power)
    plt.title(title)
    plt.xlabel('Frequenza [Hz]')
    plt.ylabel('Potenza')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_real_imag(freqs, real, imag):
    """Visualizza la parte reale e immaginaria della FFT."""
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, real, label='Parte reale')
    plt.plot(freqs, imag, label='Parte immaginaria', linestyle='dashed')
    plt.title('Parte Reale e Immaginaria della FFT')
    plt.xlabel('Frequenza [Hz]')
    plt.ylabel('Ampiezza')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def find_peak_info(freqs, power):
    """Trova informazioni sui picchi principali nello spettro."""
    # Trova i picchi sopra una certa soglia
    peaks, _ = find_peaks(power, height=np.max(power)*0.1, distance=20)
    
    # Frequenze dei picchi
    peak_freqs = freqs[peaks]
    peak_powers = power[peaks]
    
    # Picco principale
    main_peak_idx = np.argmax(peak_powers)
    main_freq = peak_freqs[main_peak_idx]
    
    # Calcola la larghezza del picco
    results_half = peak_widths(power, peaks, rel_height=0.5)
    bin_width = freqs[1] - freqs[0]
    main_width_hz = results_half[0][main_peak_idx] * bin_width
    
    # Secondo picco (se esiste)
    second_freq = None
    if len(peak_freqs) > 1:
        second_idx = np.argsort(peak_powers)[-2]
        second_freq = peak_freqs[second_idx]
    
    return main_freq, second_freq, peaks, main_peak_idx, results_half, main_width_hz


def freq_to_note(freq):
    """Converte una frequenza in nota musicale."""
    if freq == 0:
        return "N/A"
    A4 = 440.0
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    n = round(12 * np.log2(freq / A4))
    note_index = (n + 9) % 12  # A4 = index 9
    octave = 4 + (n + 9) // 12
    return f"{note_names[note_index]}{octave}"


def filter_peak(fft_result, main_freq, samplerate, width_bins=5):
    """Filtra il segnale FFT mantenendo solo il picco principale."""
    # Converti frequenza in indice nella FFT
    freq_resolution = samplerate / len(fft_result)
    main_idx = int(np.round(main_freq / freq_resolution))
    
    # Crea una maschera: finestra attorno al picco
    mask = np.zeros_like(fft_result, dtype=bool)
    mask[main_idx - width_bins : main_idx + width_bins + 1] = True
    # Per simmetria (per garantire segnale reale) considera anche la parte negativa
    mask[-main_idx - width_bins : -main_idx + width_bins + 1] = True
    
    # Applica la maschera
    filtered_fft = np.zeros_like(fft_result, dtype=complex)
    filtered_fft[mask] = fft_result[mask]
    
    # Trasformata inversa (IFFT)
    filtered_signal = np.fft.ifft(filtered_fft).real
    
    return filtered_signal


def apply_frequency_filter(audio, samplerate, filter_type="bandpass", cutoff_low=500, cutoff_high=3000):
    """Applica un filtro in frequenza al segnale audio."""
    # Calcola FFT
    fft_data = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(fft_data), d=1/samplerate)
    power_original = np.abs(fft_data)**2
    
    # Crea maschera in base al tipo di filtro
    mask = np.zeros_like(freqs, dtype=bool)
    if filter_type == "lowpass":
        mask[np.abs(freqs) <= cutoff_low] = True
    elif filter_type == "highpass":
        mask[np.abs(freqs) >= cutoff_low] = True
    elif filter_type == "bandpass":
        mask[(np.abs(freqs) >= cutoff_low) & (np.abs(freqs) <= cutoff_high)] = True
    
    # Applica maschera e calcola potenza filtrata
    filtered_fft = fft_data * mask
    power_filtered = np.abs(filtered_fft)**2
    
    # Inversa per ottenere segnale nel tempo
    filtered_signal = np.fft.ifft(filtered_fft).real
    
    return filtered_signal, freqs, power_original, power_filtered


def main():
    # 1. Leggi il file audio
    filename = 'diapason.wav'
    data, mono, samplerate = read_audio(filename)
    
    # 2. Visualizza la forma d'onda
    plot_waveform(mono, samplerate)
    
    # 3. Scrivi un nuovo file .wav identico
    sf.write('output.wav', data, samplerate)
    print("Nuovo file scritto come 'output.wav'.")
    
    # 4. Calcola FFT
    fft_result, fft_freq, power, half_freqs, half_power, half_real, half_imag = compute_fft(mono, samplerate)
    
    # 5. Visualizza spettro di potenza
    plot_power_spectrum(half_freqs, half_power)
    
    # 6. Visualizza parte reale e immaginaria
    plot_real_imag(half_freqs, half_real, half_imag)
    
    # 7. Trova e analizza i picchi
    main_freq, second_freq, peaks, main_peak_idx, results_half, main_width_hz = find_peak_info(half_freqs, half_power)
    
    # 8. Stampa informazioni sui picchi
    print(f"🎯 Frequenza principale: {main_freq:.2f} Hz")
    
    # 9. Picco secondario (se esiste)
    if second_freq:
        print(f"🎵 Picco secondario: {second_freq:.2f} Hz")
    
    # 10. Converti frequenza in nota musicale
    main_note = freq_to_note(main_freq)
    print(f"📝 Nota corrispondente al picco principale: {main_note}")
    
    if second_freq:
        second_note = freq_to_note(second_freq)
        print(f"📝 Nota secondaria (facoltativa): {second_note}")
    
    # 11. Larghezza del picco
    print(f"📏 Larghezza del picco principale: {main_width_hz:.2f} Hz")
    
    # 12. Filtra il segnale per mantenere solo il picco principale
    filtered_signal = filter_peak(fft_result, main_freq, samplerate)
    
    # 13. Salva il nuovo audio
    sf.write("solo_picco_principale.wav", filtered_signal, samplerate)
    print("✅ File 'solo_picco_principale.wav' creato con solo la frequenza dominante.")
    
    # 14. Applica un filtro in frequenza
    filter_type = "bandpass"  # "lowpass", "highpass" o "bandpass"
    cutoff_low = 500          # Hz
    cutoff_high = 3000        # solo per bandpass
    
    filtered_signal, freqs, power_original, power_filtered = apply_frequency_filter(
        mono, samplerate, filter_type, cutoff_low, cutoff_high
    )
    
    # 15. Salva audio filtrato
    sf.write(f"{filter_type}_filtrato.wav", filtered_signal, samplerate)
    print(f"✅ File '{filter_type}_filtrato.wav' salvato.")
    
    # 16. Visualizza spettri e forma d'onda filtrata
    plot_power_spectrum(freqs[:len(freqs)//2], power_original[:len(freqs)//2], 'Spettro Originale')
    plot_power_spectrum(freqs[:len(freqs)//2], power_filtered[:len(freqs)//2], 
                       f'Spettro dopo filtro {filter_type}', color='orange')
    plot_waveform(filtered_signal, samplerate, 
                 f'Segno nel tempo dopo filtro {filter_type}', color='green')


if __name__ == "__main__":
    main()