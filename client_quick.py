import os
import flwr as fl
import json
from json import JSONEncoder
import numpy as np
import tensorflow as tf
from tensorflow import keras


# for serialisation -> list to bytes, necessary for grpc (flower) communication
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# load model and data
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# define flower client
class Client(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        print(f"I´m Client {self.cid}")

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train[:10000], y_train[:10000], epochs=1, batch_size=128)
        return model.get_weights(), len(x_train[:10000]), {"cid": self.cid}

    def evaluate(self, parameters, config):
        # evaluate set with all clients
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)

        # deserialization config received from server
        missing_client = json.loads(config["missing_client"])
        decoded_weights = json.loads(config["agg_weights"])

        # evaluate all subsets without client i
        acc_dict = {"accuracy": accuracy, "missing_client": [], "accuracy_loo": []}
        for i in range((len(decoded_weights["array"]))):
            # convert received weights into a readable tensorflow format (shape)
            final_numpy_array = []
            for r in range((len(decoded_weights["array"][i]))):
                final_numpy_array.append(np.asarray(decoded_weights["array"][i][r], dtype="float32"))

            # evaluate on test data
            model.set_weights(final_numpy_array)
            loss_loo, accuracy_loo = model.evaluate(x_test, y_test)

            # add evaluate results to dict
            acc_dict["missing_client"].append(missing_client[i])
            acc_dict["accuracy_loo"].append(accuracy_loo)
        #print(acc_dict)

        # Serialization for sending back the acc_dict to server
        missing_client_ser = json.dumps(acc_dict["missing_client"]).encode('utf-8')
        accuracy_loo_ser = json.dumps(acc_dict["accuracy_loo"]).encode('utf-8')
        return loss, len(x_test), {"accuracy": accuracy, "missing_client": missing_client_ser, "accuracy_loo": accuracy_loo_ser }


# start flower client
fl.client.start_numpy_client(server_address="localhost:8080", client=Client(2))
