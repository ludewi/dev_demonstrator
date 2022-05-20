# setup
from tabnanny import verbose
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
import st_result as to_result
import matplotlib.pyplot as plt

# for caputring stdout
import contextlib
import io


def train(x_train, x_test, y_train, y_test):
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # the data, split between train and test sets


    # Scale images to the [0, 1] range
    #x_train = x_train.astype("float32") / 255
    #x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    
    captured_output = io.StringIO()

    with contextlib.redirect_stdout(captured_output):
        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    batch_size = 128
    epochs = 10

    captured_output_1 = io.StringIO()

    with contextlib.redirect_stdout(captured_output_1):
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    
   

    #print(np.shape(model.get_weights()))
    #print(model.get_weights())
    
    #input.local_weights = model.get_weights()
    
    #plt.hist(dataset=weights)
    model.save("local_model")
    #print(history.history["accuracy"])
    
    captured_string = captured_output.getvalue()
    to_result.local_train_ouput = captured_output_1.getvalue()
    
    score = model.evaluate(x_test,y_test, verbose=0)
    #history.history["accuracy"]
    return score[1]
   
#def predict(image):
#    prediction = train.model.predict(image)
#    return prediction

if __name__ == "__main__":
    train()
