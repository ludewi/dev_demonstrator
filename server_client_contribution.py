# implement new strategy
import flwr as fl
import numpy as np
import itertools
import tensorflow as tf
from tensorflow import keras
import json
from json import JSONEncoder
from typing import List, Optional
from flwr.server.strategy.aggregate import aggregate
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


# auxiliary methods
# get list of all clients who participate in federated training
def get_client_ids(metric_list):
    client_id = []
    for i in range(len(metric_list)):
        client_id.append(metric_list[i]["cid"])
    return client_id


# get all client subset combination if one client does not participate in federated training
def client_combinations(results_list):
    combinations_of_clients = []
    for i in range(len(results_list)):
        for combination in itertools.combinations(results_list, i):
            if len(combination) == len(results_list) - 1:
                combinations_of_clients.append(combination)
    return combinations_of_clients


# get all possible subsets of clients as a list
def get_subset_client_list(results_list):
    subset = []
    for r in range(len(results_list)):
        subset.append(results_list[r][2]["cid"])
    return subset


# find client id, which is not in a specific subset
def find_missing_client(all_clients, subset_clients):
    for i in range(len(all_clients)):
        if all_clients[i] not in subset_clients:
            return all_clients[i]


# for serialisation -> list to bytes, necessary for grpc (flower) communication
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# model for centralized evaluation
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


# global variables
results_fit = []
results_eval = []
aggregated_weights_dict = {"agg_weights": [], "missing_client": []}


# create new strategy -> client contribution strategy
class ClientContributionStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        number_of_clients = len(results)# Number of all participating clients

        # receive all data send back from the clients
        var_metrics = [(fit_res.metrics) for _, fit_res in results]
        var_cid = get_client_ids(var_metrics)
        var_par_num_cid = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics) for _, fit_res in results]

        # calculation of all combinations, if a client is not considered
        combinations = client_combinations(var_par_num_cid)

        # Calculation of the weights of all combinations
        for i in range(len(combinations)):
            combinations_reduced = combinations[i]
            weights_results = []
            # Read out the parameters and num_examples
            for r in range(len(combinations[i])):
                weights_results.append([combinations_reduced[r][0], combinations_reduced[r][1]])
            # return which client was not considered for aggregation
            subset_client = get_subset_client_list(combinations_reduced)
            missing_client = find_missing_client(var_cid, subset_client)

            # calculate aggregated weights and store results in dict
            aggregated_weights_loo = aggregate(weights_results)
            aggregated_weights_dict["agg_weights"].append(aggregated_weights_loo)
            aggregated_weights_dict["missing_client"].append(missing_client)

        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        return aggregated_weights

    # client-side(federated) evaluation
    def aggregate_evaluate(self, rnd, results, failures):

        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weighted accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(
            f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}"
        )

        # Call aggregate_evaluate from base class (FedAvg)
        params, _ = super().aggregate_evaluate(rnd, results, failures)

        # Calculation of Client Contribution for one round
        print("Calculation of the client contribution")
        # deserialization of weights and clients information
        missing_clients = []
        accuracies_loo = []
        for _, r in results:
            missing_cl = r.metrics["missing_client"]
            missing_clients.append(json.loads(missing_cl))
            accuracies = r.metrics["accuracy_loo"]
            accuracies_loo.append(json.loads(accuracies))

        # calculation of client contribution for one round
        def client_contribution(acc_all_clients, acc_missing_client):
            cc = (acc_all_clients-acc_missing_client)/(sum(acc_all_clients-acc_missing_client[i]))


        return params, {"accuracy": accuracy_aggregated}

    # server-side evaluation
    def get_evaluate_fn(model):
        def evaluate(self,rnd, parameters, config):
            # evalaute for set with all Clients
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)

            # evalaute for all subsets without client i
            missing_client = aggregated_weights_dict["missing_client"]
            agg_weights = aggregated_weights_dict["agg_weights"]

            # convert received weights into a readable tensorflow format (form)
            acc_dict = {"accuracy": accuracy, "missing_client": [], "accuracy_loo": []}
            for i in range((len(agg_weights))):
                final_numpy_array = []
                for r in range((len(agg_weights[i]))):
                    final_numpy_array.append(np.asarray(agg_weights[i][r]))
                model.set_weights(final_numpy_array)
                loss_loo, accuracy_loo = model.evaluate(x_test, y_test)
                acc_dict["missing_client"].append(missing_client[i])
                acc_dict["accuracy_loo"].append(accuracy_loo)
            return acc_dict

            # Calculation of Client Contribution


def evaluate_config(rnd):
    rnd = rnd
    # Serialization
    json_missing_client = json.dumps(aggregated_weights_dict["missing_client"]).encode('utf-8')
    numpy_agg_weights = np.array(aggregated_weights_dict["agg_weights"], dtype=object)
    numpy_data = {"array": numpy_agg_weights}
    json_agg_weights = json.dumps(numpy_data, cls=NumpyArrayEncoder).encode('utf-8')

    # config dict
    config = {"agg_weights": json_agg_weights,
              "missing_client": json_missing_client}
    return config


if __name__ == "__main__":

    # Initialize strategy
    strategy = ClientContributionStrategy(min_fit_clients=2,
                                          min_available_clients=2,
                                          on_evaluate_config_fn=evaluate_config)

    # Start server
    fl.server.start_server(
        server_address="localhost:8080",
        strategy=strategy,
    )