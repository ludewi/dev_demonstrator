from genericpath import exists
from typing import Any, Callable, Dict, List, Optional, Tuple, final

import flwr as fl
from tensorflow import keras


def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    # Load and compile Keras model



    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer="adam", 
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])

    # Save Server Model to JSON
    global model_json
    model_json = model.to_json()
    #with open("model.json", "w") as json_file:
    #    json_file.write(model_json)

    # Loading trained Model if exists
    #if exists('model_trained') == True:
    #   model = keras.models.load_model('model_trained')
    #else:
    #    print('Training a New Model')


    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_eval=1,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        # eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
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
        model.save('model_trained')
        return loss, {"accuracy": accuracy}

    return evaluate



def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    #json_file = open('model.json', 'r')
    #loaded_model_json = json_file.read()
    loaded_model_json = model_json
    print('Model Architecture Send To Client(s)')

    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 5,
        "model": loaded_model_json,
        "optimizer": "adam", 
        "loss": "sparse_categorical_crossentropy",
        "metrics": "accuracy",
        "val_steps": 5

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
    try:
        while True:
            main()
    except KeyboardInterrupt:
        pass
