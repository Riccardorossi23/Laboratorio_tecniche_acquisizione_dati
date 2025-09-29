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


def plot_waveform(audio, samplerate, title='Waveform (1° canale)'):
    """Visualizza la forma d'onda del segnale audio."""
    time = np.arange(len(audio)) / samplerate
    plt.figure(figsize=(10, 4))
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
    
    # Solo metà dello spettro (frequenze positive)
    half_freqs = fft_freq[:len(fft_freq)//2]
    
    # Calcola potenza, parte reale e immaginaria
    power = np.abs(fft_result)**2
    half_power = power[:len(power)//2]
    real = np.real(fft_result)[:len(fft_result)//2]
    imag = np.imag(fft_result)[:len(fft_result)//2]
    
    return half_freqs, half_power, real, imag


def plot_power_spectrum(freqs, power, title='Spettro di Potenza (FFT)'):
    """Visualizza lo spettro di potenza."""
    plt.figure(figsize=(10, 4))
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
    
    # Ordina i picchi per potenza
    sorted_indices = np.argsort(peak_powers)[::-1]
    peak_freqs = peak_freqs[sorted_indices]
    peak_powers = peak_powers[sorted_indices]
    peaks = peaks[sorted_indices]
    
    # Ottieni informazioni sul picco principale
    main_freq = peak_freqs[0]
    main_peak_idx = 0  # Ora è il primo perché abbiamo ordinato
    
    # Calcola la larghezza del picco
    results_half = peak_widths(power, [peaks[main_peak_idx]], rel_height=0.5)
    bin_width = freqs[1] - freqs[0]
    main_width_hz = results_half[0][0] * bin_width
    
    # Range min-max del picco principale
    left_idx = int(results_half[2][0])
    right_idx = int(results_half[3][0])
    min_freq = freqs[left_idx]
    max_freq = freqs[right_idx]
    
    # Secondo picco (se esiste)
    second_freq = peak_freqs[1] if len(peak_freqs) > 1 else None
    
    return main_freq, second_freq, peaks, min_freq, max_freq, main_width_hz


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


def plot_power_with_peaks(freqs, power, main_freq, second_freq, min_freq, max_freq):
    """Visualizza lo spettro di potenza con i picchi evidenziati."""
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, power, label='Spettro di Potenza')
    plt.title('Spettro di Potenza con Picchi')
    plt.xlabel('Frequenza [Hz]')
    plt.ylabel('Potenza')
    plt.grid(True)
    
    # Marker sul picco principale
    main_idx = np.argmin(np.abs(freqs - main_freq))
    plt.plot(main_freq, power[main_idx], 'rx', label=f'Picco principale ({main_freq:.2f} Hz)')
    
    # Marker sul picco secondario (se esiste)
    if second_freq:
        second_idx = np.argmin(np.abs(freqs - second_freq))
        plt.plot(second_freq, power[second_idx], 'gx', label=f'Picco secondario ({second_freq:.2f} Hz)')
    
    # Range min-max
    plt.axvline(min_freq, color='orange', linestyle='--', label=f'Range: {min_freq:.2f} Hz')
    plt.axvline(max_freq, color='orange', linestyle='--')
    
    plt.legend()
    plt.tight_layout()
    plt.show()


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
    freqs, power, real, imag = compute_fft(mono, samplerate)
    plot_power_spectrum(freqs, power)
    plot_real_imag(freqs, real, imag)
    
    # 5. Trova e analizza i picchi
    main_freq, second_freq, peaks, min_freq, max_freq, main_width_hz = find_peak_info(freqs, power)
    
    # 6. Stampa informazioni sui picchi
    print(f"🎯 Frequenza principale: {main_freq:.2f} Hz")
    main_note = freq_to_note(main_freq)
    print(f"📝 Nota corrispondente al picco principale: {main_note}")
    
    if second_freq:
        print(f"🎵 Picco secondario: {second_freq:.2f} Hz")
        second_note = freq_to_note(second_freq)
        print(f"📝 Nota secondaria: {second_note}")
    
    print(f"📏 Larghezza del picco principale: {main_width_hz:.2f} Hz")
    print(f"📊 Range di frequenze del picco principale: {min_freq:.2f} Hz – {max_freq:.2f} Hz")
    
    # 7. Visualizza lo spettro con picchi
    plot_power_with_peaks(freqs, power, main_freq, second_freq, min_freq, max_freq)


if __name__ == "__main__":
    main()