import neuralnetwork
import dataframe_create
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

def neural_network(x_train, y_train, x_test, y_test):

 '''
 print(X_train)

 print("--------------------------------------------------")
 #print(df['word_count'].head(25))
 print(df)
 print("--------------------------------------------------")
 print(len(df))
 print("--------------------------------------------------")
 '''
 model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(2)    
 ])

 predictions = model(x_train[:1]).numpy()

 tf.nn.softmax(predictions).numpy()

 loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

 loss_fn(y_train[:1], predictions).numpy()

 model.compile(optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy'])

 print(model.summary())

 # poche epoche siccome ci sono troppi pochi dati
 model.fit(x_train, y_train, epochs=10, validation_split=0.2)

 model.evaluate(x_test,  y_test, verbose=2)

 probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
 ])

 #predizioni con rete neurale
 filename2 = "predizioni"
 file_author2 = "predizioni_autori"
 df2 = dataframe_create.createdataset2test(filename2, file_author2)

 X_new = df2.values
  
 #X_new = (X_new - X_min)/(X_max-X_min)
 X_new = (X_new - X_min)/ X_max_min_diff

 input_prediction = model.predict(X_new)

 input_pred_label = np.argmax(input_prediction, axis=-1)

 print("Predizioni:")
 print(input_pred_label)

def decision_tree(x_train, y_train, x_test, y_test):
 # TODO function decision tree +  tuning (grid search cross validation)
 print("Risultati decision tree:")
 dt = DecisionTreeClassifier()
 dt.fit(x_train, y_train)
 print(dt.score(x_train, y_train), dt.score(x_test, y_test))
 # TODO plot delle caratteristiche piu importanti ()
 plot_tree(dt, filled=True, rounded = True, proportion = True)
 plt.show()

def random_forest(x_train, y_train, x_test, y_test):
 print("Risultati random forest:")
 rf = RandomForestClassifier()
 rf.fit(x_train, y_train)
 print(rf.score(x_train, y_train), rf.score(x_test, y_test))
 # dal grafico si vede l'importanza di una feature (i tag non servono quasi a nulla)
 # TODO vedere la correlazione delle feature
 plt.bar(range(0,X_train.shape[1]), rf.feature_importances_)
 plt.show()

def svm(x_train, y_train, x_test, y_test):
 print("Risultati support vector machine:")
 svm = SVC()
 svm.fit(x_train, y_train)
 print(svm.score(x_train, y_train), svm.score(x_test, y_test))

##################################### MAIN ###############################################
if __name__=='__main__':

  filename = "poesie"
  file_author = "autori"
  csv_file = "dataframe.csv"

  # Se esiste il file csv con i dati del dataset lo apre, altrimenti lo crea
  if os.path.isfile("./"+csv_file):
    df = pd.read_csv(csv_file)
  else:
   df = dataframe_create.createdatasettest(filename, file_author)
   df.to_csv("dataframe.csv", index=False)
 

  # split dei dati
  X = df.drop("author", axis=1).values
  y = df["author"].values
      
  X_train, X_test, Y_train, Y_test  = train_test_split(X, y, test_size=0.3)

  X_max = X_train.max(axis=0)
  X_min = X_train.min(axis=0)

  #print("XMAX")
  #print(X_max)
  #print("XMIN")
  #print(X_min)
  #X_train = (X_train - X_min)/(X_max-X_min)
  #X_test = (X_test - X_min)/(X_max-X_min)

  X_max_min_diff = X_max - X_min
  X_max_min_diff[X_max_min_diff == 0] = 1   # sostituisci i valori zero con 1
  X_train = (X_train - X_min) / X_max_min_diff
  X_test = (X_test - X_min)/  X_max_min_diff


  #rete neurale
  neural_network(X_train, Y_train, X_test, Y_test)
  
  #decision tree
  decision_tree(X_train, Y_train, X_test, Y_test)

  #random forest
  random_forest(X_train, Y_train, X_test, Y_test)
  
  #svm
  svm(X_train, Y_train, X_test, Y_test)
 