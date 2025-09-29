import numpy as np

class tvacTest:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.headers = None

        self._load_data()

    def _load_data(self):
        # Usa numpy per caricare i dati, saltando le righe di commento
        # Usa 'skip_header' per compatibilità con le versioni più vecchie di NumPy
        full_data = np.genfromtxt(self.filepath, delimiter='\t', skip_header=155, dtype=str)

        # Assumendo che la prima riga dei dati caricati contenga le intestazioni
        self.headers = full_data[0, :]
        self.data = full_data[1:, :] # Rimuovi la riga di intestazione dai dati

        # Assegna ogni colonna come attributo dell'istanza usando le intestazioni corrette
        for i, header_name in enumerate(self.headers): # <-- Modifica qui!
            # Assicurati che il nome dell'intestazione sia una stringa pulita
            # e valida per un nome di attributo (es. senza spazi o caratteri speciali)
            clean_header_name = header_name.strip() # Rimuovi spazi bianchi
            setattr(self, clean_header_name, self.data[:, i])

    @classmethod
    def from_file(cls, filepath):
        return cls(filepath)

    def get_column(self, name):
        return getattr(self, name, None)

    def summary(self):
        print("File:", self.filepath)
        print("Intestazioni:", self.headers)
        for h in self.headers:
            clean_h = h.strip() # Pulisci anche qui l'intestazione per coerenza
            values = getattr(self, clean_h, None) # Usa None come default se l'attributo non esiste

            if values is not None:
                try:
                    numeric_values = values.astype(float)
                    print(f"{clean_h}: min={np.min(numeric_values):.3f}, max={np.max(numeric_values):.3f}, media={np.mean(numeric_values):.3f}")
                except ValueError:
                    print(f"{clean_h}: Impossibile calcolare min/max/media (dati non numerici o non convertibili)")
            else:
                print(f"{clean_h}: Colonna non trovata o non assegnata.")


# Esempio di utilizzo (assicurati che il percorso del tuo file sia corretto)
tvac = tvacTest.from_file('/Users/riccardorossi/Desktop/laboratorio_acqu_dati/ese3_python/UTTPS_QM_20180731_restart1.asc')

# Accesso diretto a una colonna (es: 'Time' se è un'intestazione)
# print(tvac.Time)

# Riepilogo delle statistiche
tvac.summary()