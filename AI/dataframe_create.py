import preprocessing
import pandas as pd
import numpy as np


def createdataset(filename, file_author):
    word_count = preprocessing.word_count("./" + filename + ".txt")
    sentence_count = preprocessing.sentence_count("./" + filename + ".txt")
    comma_count = preprocessing.comma_count("./" + filename + ".txt")
    exclamation_count = preprocessing.exclamation_count("./" + filename + ".txt")
    unique_word_count = preprocessing.unique_word_count2("./" + filename + ".txt")
    #mean_words_phrases = preprocessing.mean_words_phrases(word_count, sentence_count)ù
    pos = preprocessing.postag_count("./" + filename + ".txt")
    author = preprocessing.author("./" + file_author + ".txt")
    #df = pd.DataFrame(list(zip(author,word_count,sentence_count, comma_count, exclamation_count, unique_word_count, mean_words_phrases)), columns = ['author','word_count','sentence_count', 'comma_count','exclamation_count','unique_word_count','mean_words_phrases'])
    df = pd.DataFrame(list(zip(author,word_count,sentence_count, comma_count, exclamation_count, unique_word_count, pos[8], pos[16])), columns = ['author','word_count','sentence_count', 'comma_count','exclamation_count','unique_word_count', 'POS8', 'POS16'])
    return df


def createdataset2(filename, file_author):
    word_count = preprocessing.word_count("./" + filename + ".txt")
    sentence_count = preprocessing.sentence_count("./" + filename + ".txt")
    comma_count = preprocessing.comma_count("./" + filename + ".txt")
    exclamation_count = preprocessing.exclamation_count("./" + filename + ".txt")
    unique_word_count = preprocessing.unique_word_count2("./" + filename + ".txt")
    pos = preprocessing.postag_count("./" + filename + ".txt")
    #mean_words_phrases = preprocessing.mean_words_phrases(word_count, sentence_count)
    #df = pd.DataFrame(list(zip(word_count,sentence_count, comma_count, exclamation_count, unique_word_count, mean_words_phrases)), columns = ['word_count','sentence_count', 'comma_count','exclamation_count','unique_word_count','mean_words_phrases'])
    df = pd.DataFrame(list(zip(word_count,sentence_count, comma_count, exclamation_count, unique_word_count, pos[8], pos[16])), columns = ['word_count','sentence_count', 'comma_count','exclamation_count','unique_word_count', 'POS8', 'POS16'])
    return df