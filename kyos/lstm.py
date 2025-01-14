import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, LSTM
from tensorflow.keras.optimizers import Adam, RMSprop, SGD


def create_lstm_model(input_shape, num_classes, neurons=[40, 30], optimizer='RMSprop', learning_rate=0.0005):
    """Creates an LSTM model."""

    model = Sequential()
    model.add(LSTM(neurons[0], input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))

    for num_neurons in neurons[1:-1]:
        model.add(LSTM(num_neurons, return_sequences=True))
        model.add(Dropout(0.2))

    model.add(LSTM(neurons[-1]))  # Last LSTM layer doesn't return sequences
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation="softmax"))

    if optimizer.lower() == 'adam':
        optimizer_instance = Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'rmsprop':
        optimizer_instance = RMSprop(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        optimizer_instance = SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer}. Choose 'Adam', 'RMSprop', or 'SGD'.")

    model.compile(
        optimizer=optimizer_instance, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model