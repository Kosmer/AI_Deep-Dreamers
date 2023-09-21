import dataframe_create
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
import joblib



def print_result_neural_network(model, x_test, y_test, x_new):
   # stampa la migliore combinazione di parametri e punteggio
   print("------------------------- Risultati Rete Neurale : ------------------------------")
   model.evaluate(x_test,  y_test, verbose=2)

   '''
   print("------------------------- PREVISIONI: ------------------------------")
   input_prediction = model.predict(x_new)

   input_pred_label = np.argmax(input_prediction, axis=-1)
   print(input_pred_label)
   '''

   print("\n************************************************************************************************\n")



def print_result_decision_tree(clf, x_test, y_test, x_new):
   # stampa la migliore combinazione di parametri e punteggio
   print("------------------------- Risultati train decision tree: ------------------------------")
   print(clf.best_params_)
   print(clf.best_estimator_)
   print(clf.best_score_)
   print(classification_report(y_test, clf.predict(x_test)))
   print("------------------------- Risultati test decision tree: ------------------------------")
   print(clf.score(x_test, y_test))

   best_max_depth = clf.best_estimator_.get_params()['max_depth']

   # plot delle caratteristiche piu importanti ()
   #plot_tree(dt, filled=True, rounded = True, proportion = True)
   #plt.show()

   '''
   print("------------------------- PREVISIONI: ------------------------------")
   input_prediction = clf.predict(x_new)
   print(input_prediction)
   '''
   
   print("\n************************************************************************************************\n")
   
   


def print_result_random_forest(clf, x_train, x_test, y_test, x_new):
   # stampa la migliore combinazione di parametri e punteggio
   print("------------------------- Risultati train random forest: ------------------------------")
   print(clf.best_params_)
   print(clf.best_estimator_)
   print(clf.best_score_)
   print(classification_report(y_test, clf.predict(x_test)))
   print("------------------------- Risultati test random forest: ------------------------------")
   print(clf.score(x_test, y_test))

   # dal grafico si vede l'importanza di una feature (i tag non servono quasi a nulla)
   plt.bar(range(0,x_train.shape[1]), clf.best_estimator_.feature_importances_)
   plt.show()
   
   '''
   print("------------------------- PREVISIONI: ------------------------------")
   input_prediction = clf.predict(x_new)
   print(input_prediction)
   '''
   
   print("\n************************************************************************************************\n")


def print_result_svm(clf, x_test, y_test, x_new):
   # stampa la migliore combinazione di parametri e punteggio
   print("------------------------- Risultati train SVM: ------------------------------")
   print(clf.best_params_)
   print(clf.best_estimator_)
   print(clf.best_score_)
   print(classification_report(y_test, clf.predict(x_test)))
   print("------------------------- Risultati test SVM: ------------------------------")

   print(clf.score(x_test, y_test))
   
   '''
   print("------------------------- PREVISIONI: ------------------------------")
   input_prediction = clf.predict(x_new)
   print(input_prediction)
   '''
   
   print("\n************************************************************************************************\n")



if __name__=='__main__':

   # Carica il modello della rete neurale
   modello_rete_neurale = tf.keras.models.load_model("modello_rete_neurale.h5")

   # Carica il modello del decision tree
   modello_decision_tree = joblib.load("modello_decision_tree.pkl")

   # Carica il modello del random forest
   modello_random_forest = joblib.load("modello_random_forest.pkl")

   # Carica il modello del svm
   modello_svm = joblib.load("modello_svm.pkl")

   # Carica le variabili dal file .npz
   data = np.load('dati_test.npz')
   X_train = data['X_train']
   X_test = data['X_test']
   Y_test = data['Y_test']


   # Carica le variabili dal secondo file .npz
   data = np.load('x_max_min.npz')
   X_min = data['X_min']
   X_max_min_diff = data['X_max_min_diff']

   filename2 = "predizioni"
   file_author2 = "predizioni_autori"
   df2 = dataframe_create.createdataset4predict(filename2)

   X_new = df2.values
   X_new = (X_new - X_min)/ X_max_min_diff

   print("\n************************************************************************************************\n")
   print_result_neural_network(modello_rete_neurale, X_test, Y_test, X_new)
   print_result_decision_tree(modello_decision_tree, X_test, Y_test, X_new)
   print_result_random_forest(modello_random_forest, X_train, X_test, Y_test, X_new)
   print_result_svm(modello_svm, X_test, Y_test, X_new)