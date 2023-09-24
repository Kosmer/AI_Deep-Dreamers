Struttura della cartella:
Cartelle:
- dataset: sono presenti tutte le poesie suddivise per autori, comprese le generazioni di chatgpt.

File txt:
- poesie.txt: contiene la raccolta di TUTTE le poesie, sia degli autori originali che di chatgpt, sequenzialmente.

- autori.txt: file che contiene le label corrispondenti alle poesie in poesie.txt. Ci sarà un valore numerico per ogni riga (il valore rappresenta la label). Il valore i-esimo è associato alla poesia i-esima nel file poesie.txt. 0 sta per essere umano, 1 ChatGPT

- predizioni.txt: contiene una raccolta di poesie sulla quale abbiamo testato il modello successivamente.

- predizioni_autori.txt: contiene le label corrispondenti alle poesie in predizioni.txt.

File csv:
- dataframe.csv: dataframe ottenuto estraendo le caratteristiche testuali dalle poesie

File .py:
- create_models.py: si occupa della creazione dei vari modelli: rete neurale, decision tree, random forest e svm.
- dataframe_create.py: si occupa dell'estrazione delle caratteristiche testuali dalle poesie in input e crea il dataset corrispondente
- merge_all.py: si occupa di unire tutte le poesie presenti nel dataset in un unico file poesie.txt e generare le label corrispondenti in un altro file autori.txt
- preprocessing.py: si occupa dell'implementazione dei metodi che si occuperanno di estrarre le caratteristiche testuali dalle poesie
- run.py: si occupa del caricamento dei modelli creati precedentemente e la stampa dei risultati ottenuti da essi.

File .pkl e .h5:
- modello_rete_neurale.h5: modello della Rete Neurale generato tramite create_models.py
- modello_decision_tree.pkl: modello del Decision Tree generato tramite create_models.py
- modello_random_forest.pkl: modello della Random Forest generato tramite create_models.py
- modello_svm.pkl: modello del SVM generato tramite create_models.py

File .npz:
- dati_test.npz: valori di train e test memorizzati per uso successivo
- x_max_min.npz: valori relativi ai modelli memorizzati per uso successivo


