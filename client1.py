from pathlib import Path
import numpy as np
import flwr as fl
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import st_result as res
import st_input as input

# model visualisation
from keras.utils.vis_utils import plot_model
import visualkeras
from keras_sequential_ascii import keras2ascii

# for caputring stdout
import contextlib
import io

hist_var = []


#counter_round = 0

def train(x_train, x_test, y_train, y_test):
    x_train_temp = x_train
    print(np.shape(x_train))
    #counter_round = 0
    # Load and compile Keras model
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

    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self):
            print("Ich bin in Parameters")
            with open('parameter.txt', 'w') as f:
                f.write('parameter')
            input.test= 5
            input.para = True
            #counter_round = 0
            #plot_model(model, to_file="get_model.png", show_shapes=True, show_layer_names=True)
            input.global_weights = model.get_weights()
            return model.get_weights()

        def fit(self, parameters, config):  # recieves the weights, and the config object what we have definded in server
            print("Ich bin in Fit")
            with open('fit.txt', 'w') as f:
                f.write('fit')
            input.fit = True
            #counter_round = counter_round + 1
            #input.counter_round = counter_round
            model.set_weights(parameters)   # the parameters which are recieved are loaded on the model schema
            r = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1) # training des Models mit den lokalen Daten des Clients
            model.save("fit_global_model")
            #plot_model(model, to_file="fit_model.png", show_shapes=True, show_layer_names=True)
            input.model_vis=visualkeras.layered_view(model)
            input.model_vis2=keras2ascii(model)
            keras2ascii(model)
            #print(model.get_weights())
            hist = r.history # the returned hist object holds a record of the loss values and metric values during training
            #print("Fit history : ", hist)
            
            hist_var.append(hist["accuracy"][-1])
            #print(hist_var)
            res.fed_hist = hist_var
            input.local_weights = model.get_weights()
            
            return model.get_weights(), len(x_train), {}    # .get_weights() - return the current model weights
                                                            # len(x_train) is the number of samples on which the model has been trained on
            # {} empty diconary weil mein Keine daten an den Server zurück senden möchte, but we you pass back any type of data
            # e.g. Authentication Tokens, client ID

        def evaluate(self, parameters, config): # is called by the server, for analytics
            print("Ich bin in Eval")
            with open('eval.txt', 'w') as f:
                f.write('eval')
            input.eval = False
            model.set_weights(parameters)
            model.save("eval_global_model")
            loss, accuracy = model.evaluate(x_test, y_test, verbose=0) #loss, accurcay auf den Testdaten
            #hist_var.append(accuracy)
            #res.fed_hist = hist_var
            print("Eval accuracy : ", accuracy) #for logging reasons
            return loss, len(x_test), {"accuracy": accuracy}    # used by the server for analytics purposes

    ######################################## Start the Numpy Client #########################################
    captured_output = io.StringIO()

    #with contextlib.redirect_stdout(captured_output):
        # every client has its own start_numpy_client, it takes in three arguments
    fl.client.start_numpy_client(
        server_address="localhost:8080",   #1 Server address, it is the endpoint which it has to hit to get the training data
                                                        # has to be the same as for the server
        client=FlowerClient(),                          # Client Object, Object of the flower client
        grpc_max_message_length=1024*1024*1024       # has to be the same lenght as the server
)

    #res.fed_train_ouput = captured_output.getvalue()

  