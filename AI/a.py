import spacy
import preprocessing


postag = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
postagcont = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Carica il modello per l'italiano
nlp = spacy.load("it_core_news_sm")

# Definisci il testo da analizzare
text = "Il cane corre nel parco. Il gatto dorme sulla finestra."

# Esegui l'analisi sintattica sul testo italiano
doc = nlp(text)

# Stampa le informazioni sintattiche del testo
for token in doc:
    for i in range(0, len(postag)):
        if token.pos_==postag[i]:
            postagcont[i]+=1
            #print(token.text, token.pos_, token.dep_)


for i in range(0,len(postag)):
    if postagcont[i]!=0:
        print("TAG "+postag[i]+" = " + str(postagcont[i]))




print(preprocessing.postag_count("./cane.txt"))