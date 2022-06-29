import json
import os
from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import flwr as fl
from sklearn.utils import shuffle
import streamlit as st


# Load dataset
arr = np.load("image_data.npz")

# save image data in variable and convert from numpy array into python array
# only then appending works with different dimension arrays eg. (5,28,28).append(1,28,28)
np_images = arr["x"]
np_x_train= np_images#.tolist()
np_y_train = arr["y"]
np_y_train = np_y_train#.tolist()



print(np_x_train.shape)
# Skalieren der Daten
x_train_norm = []
for i in range(len(np_x_train)):
    x_train_norm.append(np_x_train[i] / 255)

# reshaping the Data
x_train = np.array(x_train_norm).reshape(-1, 28, 28, 1)

# shuffle  data

X = x_train
y = np_y_train



# test train split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

x_train, y_train, x_test, y_test = x_train, y_train, x_test, y_test


# define number of images to show
num_row = 2
num_col = 8
num= num_row*num_col
# get images
images = x_train[0:num]
labels = y_train[0:num]
# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
     ax = axes[i//num_col, i%num_col]
     ax.imshow(images[i], cmap='gray_r')
     ax.set_title('Label: {}'.format(labels[i]))
plt.tight_layout()
plt.show()

# specify the maximum rotation_range angle
rotation_range_val = 30
# import relevant library
from keras.preprocessing.image import ImageDataGenerator
# create the class object
datagen = ImageDataGenerator(rotation_range=rotation_range_val)
# fit the generator
datagen.fit(x_train.reshape(x_train.shape[0], 28, 28, 1))

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_test, y_test):
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
        model_arch_str = json.dumps(model_arch_dic)

        # Return a json object
        json_obj = json.loads(model_arch_str)

        print('Json in Keras Format laden')
        # Parses a JSON model configuration string and returns a model instance.
        # create a new model from the JSON specification
        loaded_model = tf.keras.models.model_from_json(json_obj)

        # Compiler wurde nicht übergeben, daher muss das Model compiliert werden
        loaded_model.compile(optimizer=config["optimizer"], loss=config["loss"], metrics=[config["metrics"]])

        self.model = loaded_model
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

# Start Flower client
client = FlowerClient( x_train, y_train, x_test, y_test)


fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=client,

)