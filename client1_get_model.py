import argparse
import json
import os
from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import tensorflow as tf

import flwr as fl
from tensorflow import keras
from pathlib import Path

import cv2
import glob
import matplotlib.image as mpimg
from sklearn.utils import shuffle

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self , x_train, y_train, x_test, y_test):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

    #########################################################################################################   
        
        # Load new model architecture as dic
        print('Model aus dem Dic von Config vom Server laden')
        model_arch_dic = config["model"]
        # Convert dic to str object
        model_arch_str= json.dumps(model_arch_dic)

        # Return a json object
        json_obj = json.loads(model_arch_str)

        print('Json in Keras Format laden')
        # Parses a JSON model configuration string and returns a model instance.
        # create a new model from the JSON specification
        loaded_model = tf.keras.models.model_from_json(json_obj)

        # Compiler wurde nicht übergeben, daher muss das Model compiliert werden
        loaded_model.compile(optimizer=config["optimizer"], loss=config["loss"], metrics=[config["metrics"]])

        self.model =loaded_model
        # self.model.summary()    
      


    #########################################################################################################

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"Client 1: accuracy": accuracy}


def main() -> None:
    print('Client Started')
    
    # Load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Skalieren der Daten
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    # AUxillary methods
    # Visualisierung der Daten Verteilung
    def getDist(y):
        ax = sns.countplot(data = y)
        ax.set(title="Count of data classes")
        plt.show()
        plt.savefig("client1")

    # Daten aufteilen
    def getData(dist, x, y):
        dx = []
        dy = []
        counts = [0 for i in range(10)]
        for i in range(len(x)):
            if counts[y[i]]<dist[y[i]]:
                dx.append(x[i])
                dy.append(y[i])
                counts[y[i]] += 1

        return np.array(dx), np.array(dy)

    #Vorgabe der Datenverteilung
    dist = [4000, 4000, 4000, 10, 10, 10, 10, 10, 10, 10]

    # Aufrufen der Funktion getData
    x_train, y_train = getData(dist, x_train, y_train)
    # getDist(y_train)

    print('Data Loaded')
    # Start Flower client
    client = FlowerClient( x_train, y_train, x_test, y_test)

    fl.client.start_numpy_client(
        server_address="localhost:8080", 
        client=client,
        root_certificates=(
            Path(r"C:\Users\Matthias\Desktop\HS Karlsruhe\4. Semester\Masterthesis\Git\fed-learning-code\Certificate\server1.crt").read_bytes())
    )



if __name__ == "__main__":
    main()