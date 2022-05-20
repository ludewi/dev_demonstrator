from pathlib import Path
import numpy as np
import flwr as fl
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import st_result as res

hist_var = []


def load_data():
    # Load and compile Keras model
      # Load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Skalieren der Daten
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
    # defined Model schema but it does not have the weights of the server yet
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    return model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self , x_train, y_train, x_test, y_test): # warum hier die self. train 
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        return load_data().get_weights()

    def fit(self, parameters, config):  # recieves the weights, and the config object what we have definded in server
        load_data().set_weights(parameters)   # the parameters which are recieved are loaded on the model schema
        r = load_data().fit(self.x_train, self.y_train, epochs=1, validation_data=(self.x_test, self.y_test), verbose=0) # training des Models mit den lokalen Daten des Clients
        hist = r.history # the returned hist object holds a record of the loss values and metric values during training
        #print("Fit history : ", hist)
        hist_var.append(hist["accuracy"])
        print(hist_var)
        res.fed_hist = hist_var
        return load_data().get_weights(), len(self.x_train), {}    # .get_weights() - return the current model weights
                                                        # len(x_train) is the number of samples on which the model has been trained on
        # {} empty diconary weil mein Keine daten an den Server zurück senden möchte, but we you pass back any type of data
        # e.g. Authentication Tokens, client ID

    def evaluate(self, parameters, config): # is called by the server, for analytics
        load_data().set_weights(parameters)
        loss, accuracy = load_data().evaluate(self.x_test, self.y_test, verbose=0) #loss, accurcay auf den Testdaten
        print("Eval accuracy : ", accuracy) #for logging reasons
        return loss, len(self.x_test), {"accuracy": accuracy}    # used by the server for analytics purposes

    ######################################## Start the Numpy Client #########################################
def start_client(x_train, x_test, y_train, y_test):
    # every client has its own start_numpy_client, it takes in three arguments
    fl.client.start_numpy_client(
        server_address="localhost:8080",   #1 Server address, it is the endpoint which it has to hit to get the training data
                                                        # has to be the same as for the server
        client=FlowerClient(x_train, x_test, y_train, y_test),                          # Client Object, Object of the flower client
        grpc_max_message_length=1024*1024*1024       # has to be the same lenght as the server
)
