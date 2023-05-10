import tensorflow as tf
print("TensorFlow version:", tf.__version__)


class NeuralNetwork:

    x_train = 0
    y_train = 0
    model = 0
    def __init__(self, x,y, xtest, ytest):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(7, activation='relu'),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(2)    
        ])

        x_train = x
        y_train = y
        x_test = xtest
        y_test = ytest
        predictions = model(x_train[:1]).numpy()

        tf.nn.softmax(predictions).numpy()


        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        loss_fn(y_train[:1], predictions).numpy()


        model.compile(optimizer='adam',
                    loss=loss_fn,
                    metrics=['accuracy'])


        model.fit(x_train, y_train, epochs=5)


        model.evaluate(x_test,  y_test, verbose=2)


        probability_model = tf.keras.Sequential([
            model,
            tf.keras.layers.Softmax()
        ])


        print(probability_model(x_test[:5]))
        

        

        