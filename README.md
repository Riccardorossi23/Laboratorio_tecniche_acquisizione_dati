# Laboratorio Tecniche di Acquisizione Dati

Raccolta di esercizi svolti nell'ambito del corso di **Tecniche di Acquisizione Dati**, incentrati sulla generazione e sull'analisi di segnali (onde sinusoidali, triangolari e quadre) tramite Python, con un approfondimento pratico su **FPGA**.

Gli esercizi sono organizzati per difficoltà crescente: si parte dalla generazione di base dei segnali per arrivare ad analisi via via più complete, come l'individuazione di picchi massimi/minimi e punti di stallo.

---

## 📁 Struttura del repository

```
Laboratorio_tecniche_acquisizione_dati/
├── ese1_python/    # Esercizio 1 - livello base
├── ese2_python/    # Esercizio 2 - livello intermedio
├── ese3_python/    # Esercizio 3 - livello avanzato
└── fpga/           # Documentazione/relazione sulla parte FPGA
```

### `ese1_python/`
Primo esercizio, pensato come introduzione al tema. Generazione di segnali periodici di base (onde sinusoidali, triangolari, quadre) e prime analisi sulle loro caratteristiche fondamentali.

### `ese2_python/`
Esercizio di livello intermedio. Amplia l'esercizio 1 introducendo l'analisi dei segnali generati, con particolare attenzione all'individuazione di **picchi massimi e minimi**.

### `ese3_python/`
Esercizio più avanzato del percorso. Estende le analisi precedenti includendo anche lo studio dei **punti di stallo** del segnale, per una caratterizzazione più completa dell'andamento delle onde.

### `fpga/`
Contiene la documentazione e la relazione relative alla parte del laboratorio svolta su **FPGA**, con la descrizione dell'attività condotta e dei risultati ottenuti.

---

## 🎯 Obiettivo del progetto

L'obiettivo comune a tutti gli esercizi è stato quello di studiare, implementare e generare segnali periodici (onde sinusoidali, triangolari e quadre) e di condurre analisi mirate su di essi, in particolare:

- generazione di forme d'onda tramite Python;
- individuazione di punti di picco (massimo e minimo);
- individuazione di punti di stallo del segnale;
- applicazione pratica dei concetti su piattaforma FPGA.

---

## 🛠️ Tecnologie utilizzate

- **Python** (generazione e analisi dei segnali)
- **FPGA** (attività pratica documentata nella relazione)

---

## 🚀 Come consultare il progetto

Ogni cartella `eseN_python` è autonoma e contiene gli script relativi al proprio esercizio. Per eseguirli è sufficiente clonare il repository e lanciare gli script Python presenti in ciascuna cartella:

```bash
git clone https://github.com/Riccardorossi23/Laboratorio_tecniche_acquisizione_dati.git
cd Laboratorio_tecniche_acquisizione_dati
```

La cartella `fpga` è invece pensata per la sola consultazione della documentazione/relazione allegata.

---

## 👤 Autore

**Riccardo Rossi**
Progetto realizzato nell'ambito del corso di Tecniche di Acquisizione Dati.
