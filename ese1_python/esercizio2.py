#stampo onde sinusoidali,triangolari,quadre
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parametri generali
fs = 8000  # Frequenza di campionamento in Hz
duration = 0.01  # Durata del segnale in secondi
t = np.arange(0, duration, 1/fs)  # Vettore tempo

# Frequenze da plottare
frequenze = [100, 200, 440]

# Funzione per creare i segnali
def crea_segnali(t, frequenze, tipo='sin'):
    segnali = []
    for f in frequenze:
        if tipo == 'sin':
            y = np.sin(2 * np.pi * f * t)
        elif tipo == 'tri':
            y = signal.sawtooth(2 * np.pi * f * t, 0.5)
        elif tipo == 'quad':
            y = signal.square(2 * np.pi * f * t)
        segnali.append((f, y))
    return segnali

# Crea segnali
segnali_sin = crea_segnali(t, frequenze, tipo='sin')
segnali_tri = crea_segnali(t, frequenze, tipo='tri')
segnali_quad = crea_segnali(t, frequenze, tipo='quad')

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Onda sinusoidale
for f, y in segnali_sin:
    axs[0].plot(t, y, label=f'{f} Hz')
axs[0].set_title('Onda Sinusoidale')
axs[0].legend()
axs[0].grid(True)

# Onda triangolare
for f, y in segnali_tri:
    axs[1].plot(t, y, label=f'{f} Hz')
axs[1].set_title('Onda Triangolare')
axs[1].legend()
axs[1].grid(True)

# Onda quadra
for f, y in segnali_quad:
    axs[2].plot(t, y, label=f'{f} Hz')
axs[2].set_title('Onda Quadra')
axs[2].legend()
axs[2].grid(True)

# Etichette comuni
plt.xlabel('Tempo [s]')
plt.tight_layout()
plt.show()
