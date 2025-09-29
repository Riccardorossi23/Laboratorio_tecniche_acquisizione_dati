import numpy as np
import pandas as pd # Assicurati che pandas sia importato

class tvacTest:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.headers = None
        self.dataframe = None # Aggiunto per il DataFrame

        self._load_data()

    def _load_data(self):
        # Usa numpy per caricare i dati, saltando le righe di commento
        full_data = np.genfromtxt(self.filepath, delimiter='\t', skip_header=155, dtype=str)

        # Assumendo che la prima riga dei dati caricati contenga le intestazioni
        self.headers = full_data[0, :]
        # I dati rimanenti saranno le righe successive alle intestazioni
        data_rows = full_data[1:, :]

        temp_data_dict = {}

        # Assegna ogni colonna come attributo dell'istanza E prepara per il DataFrame
        for i, header_name in enumerate(self.headers):
            clean_header_name = header_name.strip() # Rimuovi spazi bianchi
            column_data = data_rows[:, i] # Usa data_rows qui

            setattr(self, clean_header_name, column_data)
            temp_data_dict[clean_header_name] = column_data # Aggiungi al dizionario per Pandas

        # Converte i dati in float per il DataFrame, ignorando gli errori di conversione
        # e usando NaN per i valori non numerici. Gestisce anche 'Date Time'.
        for key, value in temp_data_dict.items():
            if key == 'Date Time':
                try:
                    temp_data_dict[key] = temp_data_dict[key] = pd.to_datetime(value, format="%d/%m/%y %H:%M:%S")
                except Exception as e:
                    print(f"Attenzione: Impossibile convertire la colonna '{key}' in datetime: {e}")
                    temp_data_dict[key] = value # Mantieni come stringa se la conversione fallisce
            else:
                try:
                    temp_data_dict[key] = pd.to_numeric(value, errors='coerce')
                except Exception as e:
                    print(f"Attenzione: Impossibile convertire la colonna '{key}' in numerico: {e}")
                    temp_data_dict[key] = value

        # Crea il DataFrame Pandas
        self.dataframe = pd.DataFrame(temp_data_dict)


    @classmethod
    def from_file(cls, filepath):
        return cls(filepath)

    def get_column(self, name):
        return getattr(self, name, None)

    def to_dataframe(self):
        """
        Restituisce i dati come un DataFrame Pandas.
        """
        return self.dataframe

    def summary(self):
        print("File:", self.filepath)
        print("Intestazioni:", self.headers)

        print("\n--- Statistiche delle colonne ---")
        # Usa il DataFrame per le statistiche per gestire meglio i tipi
        if self.dataframe is not None:
            for col_name in self.dataframe.columns:
                if pd.api.types.is_numeric_dtype(self.dataframe[col_name]):
                    numeric_values = self.dataframe[col_name].dropna()
                    if not numeric_values.empty:
                        print(f"{col_name}: min={np.min(numeric_values):.3f}, max={np.max(numeric_values):.3f}, media={np.mean(numeric_values):.3f}")
                    else:
                        print(f"{col_name}: Nessun dato numerico valido per le statistiche.")
                elif pd.api.types.is_datetime64_any_dtype(self.dataframe[col_name]):
                    print(f"{col_name}: Tipo dati (Pandas)={self.dataframe[col_name].dtype}")
                else:
                    print(f"{col_name}: Tipo dati (Pandas)={self.dataframe[col_name].dtype} (non numerico per statistiche standard)")
        else:
            print("DataFrame non disponibile per le statistiche.")

        if self.dataframe is not None:
            print("\n--- Anteprima del DataFrame Pandas ---")
            print(self.dataframe.head())
            print("\n--- Informazioni sul DataFrame ---")
            self.dataframe.info()
        else:
            print("\nNessun DataFrame Pandas disponibile.")

    def check_tshroud_trend(self, tolerance=0.01):
        """
        Controlla l'andamento della colonna 'T Shroud' (crescente, decrescente, stabile).

        Args:
            tolerance (float): Il margine di tolleranza per considerare la variazione "stabile".
                               Valori della differenza assoluta inferiori a questo saranno stabili.

        Returns:
            str: Una stringa che descrive l'andamento principale o i conteggi degli intervalli.
        """
        if self.dataframe is None or 'T Shroud' not in self.dataframe.columns:
            return "Errore: DataFrame non disponibile o colonna 'T Shroud' mancante."

        t_shroud_data = self.dataframe['T Shroud'].dropna() # Rimuovi NaN per l'analisi

        if t_shroud_data.empty:
            return "Nessun dato valido in 'T Shroud' per l'analisi dell'andamento."

        # Calcola la differenza tra i punti consecutivi
        differences = t_shroud_data.diff()

        # Conta gli intervalli
        increasing_count = (differences > tolerance).sum()
        decreasing_count = (differences < -tolerance).sum()
        stable_count = ((differences >= -tolerance) & (differences <= tolerance)).sum()

        total_intervals = increasing_count + decreasing_count + stable_count

        if total_intervals == 0:
            return "Nessun intervallo valido trovato per l'analisi dell'andamento di 'T Shroud'."

        # Calcola le percentuali
        inc_perc = (increasing_count / total_intervals) * 100
        dec_perc = (decreasing_count / total_intervals) * 100
        stab_perc = (stable_count / total_intervals) * 100

        # Crea un riepilogo testuale
        report = (
            f"Analisi dell'andamento di 'T Shroud' (tolleranza={tolerance}):\n"
            f"  Intervalli crescenti: {increasing_count} ({inc_perc:.2f}%)\n"
            f"  Intervalli decrescenti: {decreasing_count} ({dec_perc:.2f}%)\n"
            f"  Intervalli stabili: {stable_count} ({stab_perc:.2f}%)\n"
            f"  Totale intervalli analizzati: {total_intervals}\n"
        )

        # Puoi anche aggiungere una conclusione generale
        if inc_perc > dec_perc and inc_perc > stab_perc:
            report += "L'andamento generale di 'T Shroud' è prevalentemente **crescente**.\n"
        elif dec_perc > inc_perc and dec_perc > stab_perc:
            report += "L'andamento generale di 'T Shroud' è prevalentemente **decrescente**.\n"
        elif stab_perc > inc_perc and stab_perc > dec_perc:
            report += "L'andamento generale di 'T Shroud' è prevalentemente **stabile**.\n"
        else:
            report += "L'andamento di 'T Shroud' è misto o bilanciato.\n"

        return report


# --- Esempio di utilizzo ---
tvac = tvacTest.from_file('/Users/riccardorossi/Desktop/laboratorio_acqu_dati/ese3_python/UTTPS_QM_20180731_restart1.asc')

# Riepilogo delle statistiche e anteprima del DataFrame
tvac.summary()

# Esegui il nuovo metodo per controllare l'andamento di 'T Shroud'
trend_report = tvac.check_tshroud_trend(tolerance=0.05) # Puoi specificare una tolleranza diversa
print("\n--- Analisi Andamento T Shroud ---")
print(trend_report)