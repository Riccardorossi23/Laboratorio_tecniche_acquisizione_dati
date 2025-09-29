import numpy as np
import matplotlib.pyplot as plt

# Parametri del segnale
f0 = 1  # Frequenza fondamentale (Hz)
T = 1 / f0  # Periodo
t = np.linspace(0, 2*T, 1000)  # Tempo su due periodi
N_list = [1, 3, 5, 10, 20]  # Numero di armoniche da considerare

def square_wave_fourier(t, f0, N):
    """Calcola l'approssimazione dell'onda quadra usando N armoniche dispari"""
    x = np.full_like(t, 0.5)  # Componente continua (DC)
    for k in range(1, 2*N, 2):  # Solo armoniche dispari
        coeff = (2 / (np.pi * k)) * (-1)**((k - 1) // 2)
        x += coeff * np.cos(2 * np.pi * k * f0 * t)
    return x

# Plot dell'onda quadra approssimata con diversi numeri di armoniche
plt.figure(figsize=(12, 8))
for i, N in enumerate(N_list, 1):
    plt.subplot(len(N_list), 1, i)
    plt.plot(t, square_wave_fourier(t, f0, N), label=f'N = {N} armoniche')
    plt.ylim([-0.2, 1.2])
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.suptitle("Approssimazione dell'onda quadra tramite Serie di Fourier", y=1.02)
plt.show()
