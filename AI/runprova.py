import neuralnetwork
import dataframe_create
import numpy as np

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
    #filename = input("Inserisci il nome del file delle poesie: ")
    #file_author = input("Inserisci il nome del file degli autori: ")
    filename = "poesie"
    file_author = "autori"
    df = dataframe_create.createdataset(filename, file_author)

   
    X = df.drop("author", axis=1).values
    y = df["author"].values
    
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3)

    X_max = X_train.max(axis=0)
    X_min = X_train.min(axis=0)

    X_train = (X_train - X_min)/(X_max-X_min)
    X_test = (X_test - X_min)/(X_max-X_min)


    model = neuralnetwork.NeuralNetwork()
    model.fit(X_train, y_train, epochs=500, lr=0.01)
    result = model.evaluate(X_test, y_test)

    print("ACCURATEZZA: ")
    print(model.evaluate(X_test, y_test))

    #filename2 = input("Inserisci il nome del file delle poesie da predirre: ")
    #file_author2 = input("Inserisci il nome del file degli autori da predirre: ")
    filename2 = "predizioni"
    file_author2 = "predizioni_autori"
    df2 = dataframe_create.createdataset2(filename2, file_author2)

    X_new = df2.values

    X_new = (X_new - X_min)/(X_max-X_min)

    y_pred, y_proba = model.predict(X_new, return_proba=True)

   
    classes = ["A", "B"]
    
    result = []
    with open("./" + file_author2 + ".txt", 'r', encoding="utf8") as file:
        text = file.read()
        ntext = text.split()
        for i in range(0, len(ntext)):
            result.append(int(ntext[i])) 
    
    ris = []

    

    for i, (pred, proba) in enumerate(zip(y_pred, y_proba)):
        if result[i]==int(pred):
            ris.append("CORRETTO")
        else:
            ris.append("SBAGLIATO")
        print("Risultato %d = %s (%.4f) --- %s" % (i+1, classes[int(pred)], proba, ris[i]))


    




