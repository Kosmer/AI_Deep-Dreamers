import neuralnetwork
import dataframe_create
import numpy as np
import tensorflow as tf

def train_test_split(X, y, test_size=0.3, random_state=None):

  if(random_state!=None):
    np.random.seed(random_state)
  
  n = X.shape[0]

  test_indices = np.random.choice(n, int(n*test_size), replace=False) # selezioniamo gli indici degli esempi per il test set
  
  # estraiamo gli esempi del test set
  # in base agli indici
  
  X_test = X[test_indices]
  y_test = y[test_indices]
  
  # creiamo il train set
  # rimuovendo gli esempi del test set
  # in base agli indici
  
  X_train = np.delete(X, test_indices, axis=0)
  y_train = np.delete(y, test_indices, axis=0)

  return (X_train, X_test, y_train, y_test )



if __name__=='__main__':
  filename = "poesie"
  file_author = "autori"
  df = dataframe_create.createdatasettest(filename, file_author)
    
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
      tf.keras.layers.Dense(30, activation='relu'),
      tf.keras.layers.Dense(30, activation='relu'),
      tf.keras.layers.Dense(30, activation='relu'),
      tf.keras.layers.Dense(2)    
  ])

  x_train = X_train
  y_train = Y_train
  x_test = X_test
  y_test = Y_test
  predictions = model(x_train[:1]).numpy()

  tf.nn.softmax(predictions).numpy()


  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  loss_fn(y_train[:1], predictions).numpy()


  model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


  model.fit(x_train, y_train, epochs=300)


  model.evaluate(x_test,  y_test, verbose=2)


  probability_model = tf.keras.Sequential([
      model,
      tf.keras.layers.Softmax()
  ])


  
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
  