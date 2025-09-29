import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

def read_audio(filename):
    """Legge un file audio e restituisce i dati e il sample rate"""
    data, samplerate = sf.read(filename)
    # Estrai solo un canale se stereo
    mono = data[:, 0] if data.ndim > 1 else data
    return data, mono, samplerate

def plot_waveform(mono, samplerate):
    """Disegna la forma d'onda del segnale audio"""
    time = np.arange(len(mono)) / samplerate
    plt.figure(figsize=(10, 4))
    plt.plot(time, mono)
    plt.title('Waveform (1° canale)')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Ampiezza')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_fft(mono, samplerate):
    """Calcola la FFT e restituisce frequenze, potenza, parte reale e immaginaria"""
    fft_result = np.fft.fft(mono)
    fft_freq = np.fft.fftfreq(len(mono), d=1/samplerate)
    
    # Calcola potenza, parte reale e immaginaria
    power = np.abs(fft_result)**2
    real = np.real(fft_result)
    imag = np.imag(fft_result)
    
    return fft_result, fft_freq, power, real, imag

def plot_power_spectrum(fft_freq, power):
    """Disegna lo spettro di potenza"""
    plt.figure(figsize=(10, 4))
    plt.plot(fft_freq[:len(fft_freq)//2], power[:len(power)//2])
    plt.title('Spettro di Potenza (FFT)')
    plt.xlabel('Frequenza [Hz]')
    plt.ylabel('Potenza')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_real_imag(fft_freq, real, imag):
    """Disegna parte reale e immaginaria della FFT"""
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

def find_main_peaks(fft_freq, power):
    """Trova i picchi principali nello spettro di potenza"""
    # Considera solo la metà positiva dello spettro
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
    
    # Picco secondario (se esiste)
    second_freq = None
    if len(peak_freqs) > 1:
        second_idx = np.argsort(peak_powers)[-2]
        second_freq = peak_freqs[second_idx]
        print(f"🎵 Picco secondario: {second_freq:.2f} Hz")
    
    # Calcola larghezza del picco principale
    results_half = peak_widths(half_power, peaks, rel_height=0.5)
    main_width_hz = results_half[0][main_peak_idx] * (half_freqs[1] - half_freqs[0])
    print(f"📏 Larghezza del picco principale: {main_width_hz:.2f} Hz")
    
    return main_freq, second_freq, peaks, main_peak_idx, half_freqs, half_power

def freq_to_note(freq):
    """Converte una frequenza in una nota musicale"""
    if freq == 0:
        return "N/A"
    A4 = 440.0
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    n = round(12 * np.log2(freq / A4))
    note_index = (n + 9) % 12  # A4 = index 9
    octave = 4 + (n + 9) // 12
    return f"{note_names[note_index]}{octave}"

def isolate_peak(fft_result, main_freq, samplerate):
    """Isola il picco principale nello spettro"""
    # Converti frequenza in indice nella FFT
    freq_resolution = samplerate / len(fft_result)
    main_idx = int(np.round(main_freq / freq_resolution))
    
    # Crea una maschera: finestra attorno al picco
    width_bins = 5
    mask = np.zeros_like(fft_result, dtype=bool)
    mask[main_idx - width_bins : main_idx + width_bins + 1] = True
    # Per simmetria (per garantire segnale reale)
    mask[-main_idx - width_bins : -main_idx + width_bins + 1] = True
    
    # Applica la maschera
    filtered_fft = np.zeros_like(fft_result, dtype=complex)
    filtered_fft[mask] = fft_result[mask]
    
    # Trasformata inversa
    filtered_signal = np.fft.ifft(filtered_fft).real
    
    return filtered_signal

def apply_filter(mono, samplerate, filter_type="lowpass", cutoff_low=1000, cutoff_high=3000):
    """Applica un filtro nel dominio della frequenza"""
    # Calcolo FFT
    fft_data = np.fft.fft(mono)
    freqs = np.fft.fftfreq(len(fft_data), d=1/samplerate)
    
    # Crea maschera del filtro
    mask = np.zeros_like(freqs, dtype=bool)
    
    if filter_type == "lowpass":
        mask[np.abs(freqs) <= cutoff_low] = True
    elif filter_type == "highpass":
        mask[np.abs(freqs) >= cutoff_low] = True
    elif filter_type == "bandpass":
        mask[(np.abs(freqs) >= cutoff_low) & (np.abs(freqs) <= cutoff_high)] = True
    
    # Applica il filtro
    filtered_fft = fft_data * mask
    
    # Trasformata inversa
    filtered_signal = np.fft.ifft(filtered_fft).real
    
    return filtered_signal

def main():
    # 1. Leggi il file audio
    filename = 'diapason.wav'
    data, mono, samplerate = read_audio(filename)
    
    # 2. Visualizza la forma d'onda
    plot_waveform(mono, samplerate)
    
    # 3. Scrivi un nuovo file .wav identico
    sf.write('output.wav', data, samplerate)
    print("Nuovo file scritto come 'output.wav'.")
    
    # 4. Calcola e visualizza FFT
    fft_result, fft_freq, power, real, imag = calculate_fft(mono, samplerate)
    plot_power_spectrum(fft_freq, power)
    plot_real_imag(fft_freq, real, imag)
    
    # 5. Trova i picchi principali
    main_freq, second_freq, peaks, main_peak_idx, half_freqs, half_power = find_main_peaks(fft_freq, power)
    
    # 6. Converti frequenza in nota musicale
    main_note = freq_to_note(main_freq)
    print(f"📝 Nota corrispondente al picco principale: {main_note}")
    
    if second_freq:
        second_note = freq_to_note(second_freq)
        print(f"📝 Nota secondaria (facoltativa): {second_note}")
    
    # 7. Isola il picco principale
    filtered_signal = isolate_peak(fft_result, main_freq, samplerate)
    sf.write("solo_picco_principale.wav", filtered_signal, samplerate)
    print("✅ File 'solo_picco_principale.wav' creato con solo la frequenza dominante.")
    
    # 8. Applica un filtro passa-basso
    filtered_signal = apply_filter(mono, samplerate, "lowpass", 1000)
    sf.write("audio_filtrato.wav", filtered_signal, samplerate)
    print("✅ Filtro lowpass applicato. File salvato come 'audio_filtrato.wav'.")
    
    # 9. Salva il risultato del filtro sintetizzato
    sf.write("audio_filtrato_sintetizzato.wav", filtered_signal, samplerate)
    print("✅ Nuovo file audio sintetizzato creato: 'audio_filtrato_sintetizzato.wav'")

if __name__ == "__main__":
    main()