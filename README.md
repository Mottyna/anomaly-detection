# Anomaly detection
Project work per il corso di Computer Vision presso [unife](https://corsi.unife.it/it/lm-ia). 

Lo scopo del progetto era implementare e testare con vari modelli la tecnica di *knowledge distillation* per 
*anomaly detection* su datasets [mvtecAD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).


## Caratteristiche principali
* **Multi-architettura:** Supporto per diversi modelli Teacher pre-addestrati (`resnet18`, `resnet50`, `wideresnet50`).
* **Due paradigmi di distillazione:**
    * **Standard knowledge distillation:** Allineamento diretto delle feature tramite conv 1x1 (`ProjectorWrapper`).
    * **Reverse Distillation for Anomaly Detection (RD4AD):** Passaggio delle feature del Teacher attraverso un collo di bottiglia (`Bottleneck`) e successiva ricostruzione tramite Decoder con connessioni skip non corrispondenti.
* **Calibrazione automatica del threshold:** Calcolo della soglia ottimale di decisione sul set di validazione tramite la generazione di anomalie sintetiche (tecniche di *Cut-Paste* e *Alpha Blending Noise/Color*) e l'indice di Youden.
* **Valutazione:** Calcolo delle metriche di performance sia a livello immagine che a livello pixel:
    * Image-level & Pixel-level ROC-AUC
    * PR-AUC globale
    * Precision, Recall e F1-Score
* **Visualizzazione delle anomalie:** Generazione di heatmap comparative (immagine originale, maschera *ground truth* e *anomaly map*) salvate in formato PNG.


## Struttura del progetto

```text
├── main.py          # Entry point del programma (gestione argomenti, loader e pipeline)
├── classes.py       # Definizione delle reti (Teacher, Projector, RD4AD Student)
├── actions.py       # Funzioni core: train, test, calibrazione soglia e generazione anomalie sintetiche
├── mvtec.py         # Dataloader per la gestione dei flussi di Train e Test/GT
├── parameters.py    # Costanti, seed, iperparametri e pesi dei layer
├── risultati/       # [Generata automaticamente] Cartella contenente i plot di output del test
└── mvtec/           # Cartella che deve contenere i database mvtec sui quali si vuole testare il programma
```


## Requisiti e installazione
Il progetto richiede Python 3.8+ e le seguenti librerie principali:
```
pip install torch torchvision scikit-learn numpy matplotlib pillow
```

### Configurazione del dataset
Scarica il dataset MVTec AD e posizionalo all'interno di una cartella chiamata mvtec nella root del progetto. La struttura delle cartelle deve rispettare lo standard MVTec:
```
./mvtec/
└── bottle/
    ├── ground_truth/
    |   ├── broken_large/
    |   ├── broken_small/
    |   └── contamination/
    ├── test/
    │   ├── broken_large/
    |   ├── broken_small/
    |   ├── contamination/
    |   └── good/
    ├── train/
        └── good/
```


## Come utilizzare il progetto
Il file `main.py` accetta tre argomenti posizionali da riga di comando:
```
python3 main.py <categoria_dataset> <modello_teacher> <modello_student>
```

### Esempi di esecuzione:
Esegui con reverse distillation (RD4AD) su "bottle":
```
python3 main.py bottle wideresnet50 rd4ad
```
Esegui con distillazione standard su "carpet":
```
python3 main.py carpet resnet50 resnet18
```


## Dettagli tecnici rilevanti
### Prevenzione dell'effetto bordo
Nella fase di estrazione della mappa di anomalie (`actions.py`), i bordi dell'immagine tendono a generare falsi positivi. Il codice azzera artificialmente un margine di 10 pixel lungo i bordi per pulire il segnale.

### Top-k pixel pooling
Per evitare che la dimensione fisica del difetto influenzi eccessivamente lo score globale dell'immagine, la metrica a livello di immagine viene calcolata estraendo solo lo 0.5% dei pixel peggiori (con errore più alto) e facendone la media.

### Bilanciamento dei Layer (`parameters.py`)
È possibile associare pesi diversi ai vari layer della rete student nella *reverse distillation* a seconda della categoria del dataset (es. dare più importanza ai layer iniziali per le textures o ai layer profondi per le forme).



## Output
Al termine della fase di test, all'interno della cartella `risultati/` verranno salvate immagini di confronto nominate `anomaly_sample_[ID].png`. Ogni immagine contiene:
* **Originale**: L'immagine di input denormalizzata con il target reale.
* **Ground Truth**: La maschera binaria reale del difetto.
* **Anomaly Map (Heatmap)**: La mappa di calore generata dal modello (in scala di colori jet), affiancata dalla predizione finale del sistema (Normale o Anomalo).

Sul terminale verrano stampate le metriche di valutazione:
- ROC-AUC (globale e a livello di pixel)
- PR-AUC
- F1 score
- precision
- recall
- threshold ottimale calcolato a posteriori (*non* quello usato per calcolare le metriche)


## Crediti

Questo progetto include e adatta codice open-source di terze parti:

* **MVTec Dataloader (`mvtec.py`):** Adattato dal repository originale di [@b3r8](https://github.com/b3r8) ([b3r8/mvtec-dataloader](https://github.com/b3r8/mvtec-dataloader)), rilasciato sotto licenza **MIT**. Il file originale è stato modificato per integrare e restituire le maschere di Ground Truth (GT) necessarie per la valutazione a livello di pixel.




