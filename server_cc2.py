from typing import Any, Callable, Dict, List, Optional, Tuple, final

import flwr as fl
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Lambda, MaxPooling2D # convolution layers
from keras.layers import Dense, Dropout, Flatten # core layers
from keras.layers import BatchNormalization


def main() -> None:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer="adam", 
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_eval=1,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        #eval_fn= get_eval_fn(model),
        on_fit_config_fn=fit_config(),
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for five rounds of federated learning
    fl.server.start_server(
            server_address="localhost:8080",
            config={"num_rounds": 5}, 
            strategy=strategy,
            # Forces a distributed evaluation to occur after the last training epoch when enabled.
            force_final_distributed_eval = True, ########### NEU ############### app.py

    )

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss = 0
        accuracy = 0
        return loss, {"accuracy": accuracy}

    return evaluate



def fit_config():
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """

    config = {
        "batch_size": 32,
        "local_epochs": 10,
        "optimizer": "adam", 
        "loss": "sparse_categorical_crossentropy",
        "metrics": "accuracy",
        "val_steps": 5,
        "round": 1,

    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
