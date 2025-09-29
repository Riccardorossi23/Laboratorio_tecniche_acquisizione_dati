import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Importa la libreria matplotlib

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

        # Converte i dati per il DataFrame
        for key, value in temp_data_dict.items():
            # Converte 'Date Time' in datetime, altre in numerico (con errori a NaN)
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

    def plot_data(self):
        """
        Genera i plot richiesti utilizzando Matplotlib.
        """
        if self.dataframe is None or 'Date Time' not in self.dataframe.columns:
            print("Errore: DataFrame non disponibile o colonna 'Date Time' mancante.")
            return

        time_column = self.dataframe['Date Time']

        # --- Plot 1: T Shroud e T Shroud Setpoint vs Date Time ---
        # Creiamo due subplot: uno per la vista generale e uno con lo zoom sull'asse Y
        fig1, (ax1_raw, ax1_zoom) = plt.subplots(2, 1, figsize=(12, 10))
        fig1.suptitle('Temperature: T Shroud e T Shroud Setpoint vs Date Time', fontsize=16)

        # Plot su ax1_raw (vista generale)
        ax1_raw.plot(time_column, self.dataframe['T Shroud'], label='T Shroud', color='blue')
        ax1_raw.plot(time_column, self.dataframe['T Shroud Setpoint'], label='T Shroud Setpoint', color='red', linestyle='--')
        ax1_raw.set_title('Vista Generale (Range Completo)')
        ax1_raw.set_xlabel('Data e Ora')
        ax1_raw.set_ylabel('Temperatura')
        ax1_raw.legend()
        ax1_raw.grid(True)
        fig1.autofmt_xdate() # Formatta automaticamente le etichette dell'asse X

        # Plot su ax1_zoom (dettaglio con zoom sull'asse Y)
        ax1_zoom.plot(time_column, self.dataframe['T Shroud'], label='T Shroud', color='blue')
        ax1_zoom.plot(time_column, self.dataframe['T Shroud Setpoint'], label='T Shroud Setpoint', color='red', linestyle='--')
        ax1_zoom.set_title('Dettaglio (Zoom su Asse Y)')
        ax1_zoom.set_xlabel('Data e Ora')
        ax1_zoom.set_ylabel('Temperatura')
        ax1_zoom.legend()
        ax1_zoom.grid(True)
        # Calcola il range min/max per lo zoom, escludendo NaN
        temp_cols = self.dataframe[['T Shroud', 'T Shroud Setpoint']].dropna()
        if not temp_cols.empty:
            min_val = temp_cols.min().min()
            max_val = temp_cols.max().max()
            # Imposta un margine per lo zoom (es. 10% in più/meno del range effettivo)
            margin = (max_val - min_val) * 0.1
            ax1_zoom.set_ylim(min_val - margin, max_val + margin) # Imposta i limiti dell'asse Y
        fig1.autofmt_xdate()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Aggiusta il layout per evitare sovrapposizioni del titolo

        # --- Plot 2: Pressioni vs Date Time ---
        # Creiamo due subplot: uno con scala lineare e uno con scala logaritmica
        fig2, (ax2_linear, ax2_log) = plt.subplots(2, 1, figsize=(12, 10))
        fig2.suptitle('Pressioni: P Full Range 1 chamber-ITR90 e P Chamber vs Date Time', fontsize=16)

        # Plot su ax2_linear (scala lineare)
        ax2_linear.plot(time_column, self.dataframe['P Full Range 1 chamber-ITR90'], label='P Full Range 1 chamber-ITR90', color='green')
        ax2_linear.plot(time_column, self.dataframe['P Chamber'], label='P Chamber', color='purple', linestyle=':')
        ax2_linear.set_title('Vista Lineare')
        ax2_linear.set_xlabel('Data e Ora')
        ax2_linear.set_ylabel('Pressione (Lineare)')
        ax2_linear.legend()
        ax2_linear.grid(True)
        fig2.autofmt_xdate()

        # Plot su ax2_log (scala logaritmica)
        ax2_log.plot(time_column, self.dataframe['P Full Range 1 chamber-ITR90'], label='P Full Range 1 chamber-ITR90', color='green')
        ax2_log.plot(time_column, self.dataframe['P Chamber'], label='P Chamber', color='purple', linestyle=':')
        ax2_log.set_title('Vista Logaritmica')
        ax2_log.set_xlabel('Data e Ora')
        ax2_log.set_ylabel('Pressione (Logaritmica)')
        ax2_log.set_yscale('log') # Imposta la scala logaritmica sull'asse Y
        ax2_log.legend()
        ax2_log.grid(True, which="both", ls="-", lw=0.5) # Griglia per scala logaritmica
        fig2.autofmt_xdate()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show() # Mostra tutte le figure

    def plot_tshroud_trend(self, tolerance=0.01):
        """
        Disegna 'T Shroud' vs 'Date Time' indicando gli intervalli in cui è crescente,
        decrescente o stabile.

        Args:
            tolerance (float): Il margine di tolleranza per considerare un intervallo "stabile".
                               Valori della differenza assoluta inferiori a questo saranno stabili.
        """
        if self.dataframe is None or 'Date Time' not in self.dataframe.columns or 'T Shroud' not in self.dataframe.columns:
            print("Errore: DataFrame non disponibile o colonne 'Date Time' o 'T Shroud' mancanti.")
            return

        time_data = self.dataframe['Date Time']
        t_shroud_data = self.dataframe['T Shroud']

        # Rimuovi eventuali righe con valori NaN in 'T Shroud' per l'analisi della derivata
        df_clean = self.dataframe[['Date Time', 'T Shroud']].dropna(subset=['T Shroud']).copy()
        time_clean = df_clean['Date Time']
        t_shroud_clean = df_clean['T Shroud']

        if t_shroud_clean.empty:
            print("Nessun dato valido per 'T Shroud' dopo la pulizia dei NaN.")
            return

        # Calcola la differenza tra i punti consecutivi
        differences = t_shroud_clean.diff()

        # Inizializza le liste per i segmenti colorati
        increasing_x, increasing_y = [], []
        decreasing_x, decreasing_y = [], []
        stable_x, stable_y = [], []

        # Itera sui dati per classificare e preparare i segmenti per il plot
        for i in range(1, len(differences)):
            # Prendi i punti attuali e precedenti
            x_prev, y_prev = time_clean.iloc[i-1], t_shroud_clean.iloc[i-1]
            x_curr, y_curr = time_clean.iloc[i], t_shroud_clean.iloc[i]
            diff_val = differences.iloc[i]

            # Classifica l'intervallo
            if pd.isna(diff_val): # Salta se la differenza è NaN (es. primo punto)
                continue
            elif diff_val > tolerance:
                increasing_x.extend([x_prev, x_curr])
                increasing_y.extend([y_prev, y_curr])
                increasing_x.append(None) # Usa None per spezzare la linea
                increasing_y.append(None)
            elif diff_val < -tolerance:
                decreasing_x.extend([x_prev, x_curr])
                decreasing_y.extend([y_prev, y_curr])
                decreasing_x.append(None)
                decreasing_y.append(None)
            else: # abs(diff_val) <= tolerance
                stable_x.extend([x_prev, x_curr])
                stable_y.extend([y_prev, y_curr])
                stable_x.append(None)
                stable_y.append(None)

        # Crea il plot
        plt.figure(figsize=(14, 7))
        # Disegna i segmenti colorati. L'uso di None nelle liste crea segmenti separati.
        plt.plot(increasing_x, increasing_y, color='green', label='Crescente', linewidth=2)
        plt.plot(decreasing_x, decreasing_y, color='red', label='Decrescente', linewidth=2)
        plt.plot(stable_x, stable_y, color='blue', label='Stabile', linewidth=2)

        plt.title('T Shroud vs Date Time con Andamento', fontsize=16)
        plt.xlabel('Data e Ora')
        plt.ylabel('T Shroud')
        plt.legend()
        plt.grid(True)
        #plt.autofmt_xdate() # Formatta le etichette dell'asse X
        plt.tight_layout()
        plt.show()


# --- Esempio di utilizzo ---
tvac = tvacTest.from_file('/Users/riccardorossi/Desktop/laboratorio_acqu_dati/ese3_python/UTTPS_QM_20180731_restart1.asc')

# Riepilogo delle statistiche e anteprima del DataFrame
tvac.summary()

# Esegui il metodo per controllare l'andamento di 'T Shroud'
trend_report = tvac.check_tshroud_trend(tolerance=0.05) # Puoi specificare una tolleranza diversa
print("\n--- Analisi Andamento T Shroud ---")
print(trend_report)

# Genera e mostra i plot generali (temperature e pressioni)
tvac.plot_data()

# Genera e mostra il plot di T Shroud con gli andamenti evidenziati
tvac.plot_tshroud_trend(tolerance=0.05) # Regola la tolleranza per il plot se necessario