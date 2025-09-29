import numpy as np
import matplotlib.pyplot as plt

# Parametri comuni
fs = 8000  # Frequenza di campionamento in Hz
duration = 0.01  # Durata del segnale in secondi
t = np.arange(0, duration, 1/fs)  # Vettore del tempo

# Frequenze dei segnali
frequencies = [100, 200, 440]

# Creazione e visualizzazione dei segnali
plt.figure(figsize=(10, 6))

for f in frequencies:
    y = np.sin(2 * np.pi * f * t)
    plt.plot(t, y, label=f'{f} Hz')

plt.title('Onde sinusoidali a 100 Hz, 200 Hz e 440 Hz')
plt.xlabel('Tempo [s]')
plt.ylabel('Ampiezza')
plt.legend()
plt.grid(True)
plt.show()