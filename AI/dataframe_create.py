import preprocessing
import pandas as pd
import numpy as np


def createdataset(filename, file_author):
    word_count = preprocessing.word_count("./" + filename + ".txt")
    sentence_count = preprocessing.sentence_count("./" + filename + ".txt")
    comma_count = preprocessing.comma_count("./" + filename + ".txt")
    exclamation_count = preprocessing.exclamation_count("./" + filename + ".txt")
    unique_word_count = preprocessing.unique_word_count2("./" + filename + ".txt")
    mean_words_phrases = preprocessing.mean_words_phrases(word_count, sentence_count)
    pol_sub = preprocessing.polarity_subjectivity("./" + filename + ".txt")
    polarity = pol_sub[0]
    subjectivity = pol_sub[1]
    pos = preprocessing.postag_count("./" + filename + ".txt")
    author = preprocessing.author("./" + file_author + ".txt")

    pos0 = pos1 = pos2 = pos3 = pos4 = pos5 = pos6 = pos7 = pos8 = pos9 = pos10 = pos11 = pos12 = pos13 = pos14 = pos15 = pos16 = pos17 = pos18 = []
    for lista in pos:
        pos0.append(lista[0])
        pos1.append(lista[1])
        pos2.append(lista[2])
        pos3.append(lista[3])
        pos4.append(lista[4])
        pos5.append(lista[5])
        pos6.append(lista[6])
        pos7.append(lista[7])
        pos8.append(lista[8])
        pos9.append(lista[9])
        pos10.append(lista[10])
        pos11.append(lista[11])
        pos12.append(lista[12])
        pos13.append(lista[13])
        pos14.append(lista[14])
        pos15.append(lista[15])
        pos16.append(lista[16])
        pos17.append(lista[17])
        pos18.append(lista[18])

    df = pd.DataFrame(list(zip(author,word_count,sentence_count, comma_count, exclamation_count, unique_word_count, mean_words_phrases, polarity, subjectivity, pos0, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9, pos10, pos11, pos12, pos13, pos14, pos15, pos16, pos17, pos18)), columns = ['author','word_count','sentence_count', 'comma_count','exclamation_count','unique_word_count', 'mean_words_phrases', 'polarity', 'subjectivity','POS0','POS1','POS2','POS3','POS4','POS5','POS6','POS7','POS8','POS9','POS10','POS11','POS12','POS13','POS14','POS15','POS16','POS17','POS18'])
    
    return df


def createdataset4predict(filename):
    word_count = preprocessing.word_count("./" + filename + ".txt")
    sentence_count = preprocessing.sentence_count("./" + filename + ".txt")
    comma_count = preprocessing.comma_count("./" + filename + ".txt")
    exclamation_count = preprocessing.exclamation_count("./" + filename + ".txt")
    unique_word_count = preprocessing.unique_word_count2("./" + filename + ".txt")
    mean_words_phrases = preprocessing.mean_words_phrases(word_count, sentence_count)
    pol_sub = preprocessing.polarity_subjectivity("./" + filename + ".txt")
    polarity = pol_sub[0]
    subjectivity = pol_sub[1]
    pos = preprocessing.postag_count("./" + filename + ".txt")
    

    

    pos0 = pos1 = pos2 = pos3 = pos4 = pos5 = pos6 = pos7 = pos8 = pos9 = pos10 = pos11 = pos12 = pos13 = pos14 = pos15 = pos16 = pos17 = pos18 =[]
    for lista in pos:
        pos0.append(lista[0])
        pos1.append(lista[1])
        pos2.append(lista[2])
        pos3.append(lista[3])
        pos4.append(lista[4])
        pos5.append(lista[5])
        pos6.append(lista[6])
        pos7.append(lista[7])
        pos8.append(lista[8])
        pos9.append(lista[9])
        pos10.append(lista[10])
        pos11.append(lista[11])
        pos12.append(lista[12])
        pos13.append(lista[13])
        pos14.append(lista[14])
        pos15.append(lista[15])
        pos16.append(lista[16])
        pos17.append(lista[17])
        pos18.append(lista[18])


    df = pd.DataFrame(list(zip(word_count,sentence_count, comma_count, exclamation_count, unique_word_count, mean_words_phrases, polarity, subjectivity, pos0, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9, pos10, pos11, pos12, pos13, pos14, pos15, pos16, pos17, pos18)), columns = ['word_count','sentence_count', 'comma_count','exclamation_count','unique_word_count', 'mean_words_phrases', 'polarity', 'subjectivity', 'POS0','POS1','POS2','POS3','POS4','POS5','POS6','POS7','POS8','POS9','POS10','POS11','POS12','POS13','POS14','POS15','POS16','POS17','POS18'])
    
    return df