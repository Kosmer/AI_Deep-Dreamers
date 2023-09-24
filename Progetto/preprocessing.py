import re
import spacy
from pattern.it import sentiment


def normalize_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    stop_words = set(stopwords.words("italian"))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


#calcola il numero di parole

def word_count(filename):
    ris = []
    with open(filename, 'r', encoding="utf8") as file:
        text = file.read()
        ntext = text.split('*')
        for i in range(0, len(ntext)-1):
            words = ntext[i].split()
            ris.append(len(words))
        
    return ris



def postag_count(filename):
    nlp = spacy.load("it_core_news_sm")
    postag = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
    ris = []
    with open(filename, 'r', encoding="utf8") as file:
        text = file.read()
        ntext = text.split('*')
        for i in range(0, len(ntext)-1):
            doc = nlp(ntext[i])

            postagcont = []
            for j in range(0, len(postag)):
                postagcont.append(0)
            
            for token in doc:
                for j in range(0, len(postag)):
                    if token.pos_==postag[j]:
                        postagcont[j]+=1

            words = ntext[i].split()
            nwords = len(words)
            for j in range(0, len(postag)):
                postagcont[j] = (postagcont[j]/nwords)

            ris.append(postagcont)

        
    return ris


def word_count2(filename):
    ris = []
    with open(filename, 'r', encoding="utf8") as file:
        text = file.read()
        words = text.split() 
        return len(words)



#numero di frasi
import re
def sentence_count(filename):
    ris = []
    with open(filename, 'r', encoding="utf8") as file:
        text = file.read()
        ntext = text.split('*')
        for i in range(0, len(ntext)-1):
            sentences =  re.split(r'[.!?]+', ntext[i])
            ris.append(len(sentences)-1)
        return ris



#numero di virgole  (che si può migliorare con numero di virgole per frase)

def comma_count(filename):
    ris = []
    with open(filename, 'r', encoding="utf8") as file:
        text = file.read()
        ntext = text.split('*')
        for i in range(0, len(ntext)-1):
            comma_count = ntext[i].count(',')
            ris.append(comma_count)
    
    return ris




#numero medio di parole per frase (rapporto tra word_count/sentence_count)

def unique_word_count(filename):
    ris = []
    with open(filename, 'r', encoding="utf8") as file:
        text = file.read()
        ntext = text.split('*')
        for i in range(0, len(ntext)-1):
            words = ntext[i].split()
            unique_words = set(words)
            ris.append(len(unique_words))
    
    return ris


def mean_words_phrases(ris1, ris2):
    ris = []
    
    for i in range(0, len(ris1)):
        if ris2[i]==0:
            ris2[i] = 1
        ris.append(int(ris1[i]/ris2[i]))

    return ris



#numero di punti esclamativi

def exclamation_count(filename):
    ris = []
    with open(filename, 'r', encoding="utf8") as file:
        text = file.read()
        ntext = text.split('*')
        for i in range(0, len(ntext)-1):
            exclamation_count = ntext[i].count('!')
            ris.append(exclamation_count)
    
    return ris


#numero di parole diverse, va fare prima e dopo la normalizzazione

def unique_word_count2(filename):
    ris = []
    with open(filename, 'r', encoding="utf8") as file:
        text = file.read()
        ntext = text.split('*')
        for i in range(0, len(ntext)-1):
            words = ntext[i].split()
            word_dict = {}
            for word in words:
                if word not in word_dict:
                    word_dict[word] = 1
            
            ris.append(len(word_dict))

    return ris

# felicita della frase + quanto è sicuro che il primo valore sia corretto
def polarity_subjectivity(filename):
    ris1 = []
    ris2 = []
    with open(filename, 'r', encoding="utf8") as file:
        text = file.read()
        ntext = text.split('*')
        for i in range(0, len(ntext)-1):
            polarity = sentiment(ntext[i])[0]
            subjectivity = sentiment(ntext[i])[1]
            ris1.append(polarity)
            ris2.append(subjectivity)

    return ris1, ris2


def number(filename):
    ris = []
    with open(filename, 'r', encoding="utf8") as file:
        text = file.read()
        ntext = text.split('*')
        print(len(ntext)-1)


def author(filename):
    ris = []
    with open(filename, 'r', encoding="utf8") as file:
        text = file.read()
        ntext = text.split()
        for i in range(0, len(ntext)):
            ris.append(int(ntext[i])) 
        
        return ris


    