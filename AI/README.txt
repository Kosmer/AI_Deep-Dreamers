Struttura della cartella:
- dataset: sono presenti tutte le poesie suddivise per autori, comprese le generazioni di chatgpt.
- poesie.txt: contiene la raccolta di TUTTE le poesie, sia degli autori originali che di chatgpt, sequenzialmente.
- autori.txt: file che contiene le label corrispondenti alle poesie in poesie.txt. Ci sarà un valore numerico per ogni riga (il valore rappresenta la label). Il valore i-esimo è associato alla poesia i-esima nel file poesie.txt.
- poesie_originali.txt: contiene la raccolta di tutte le poesie originali dei vari autori.
- numero_poesie_originali.txt: contiene le label delle poesie originali. Ci sarà un valore numerico per ogni riga (il valore rappresenta la label). Il valore i-esimo è associato alla poesia i-esima nel file poesie_originali.txt.
- poesie_ChatGPT.txt: contiene la raccolta di tutte le poesie di ChatGPT dei vari autori.
- numero_poesie_ChatGPT.txt: contiene le label delle poesie di ChatGPT. Ci sarà un valore numerico per ogni riga (il valore rappresenta la label). Il valore i-esimo è associato alla poesia i-esima nel file poesie_ChatGPT.txt.
- predizioni.txt: contiene una raccolta di poesie sulla quale testare il modello una volta addestrato.
- predizioni_autori.txt: contiene le label corrispondenti alle poesie in predizioni.txt.
