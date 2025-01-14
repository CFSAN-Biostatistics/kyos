# -*- coding: utf-8 -*-

"""
test_lstm
----------------------------------

Tests for `lstm` module.
"""

from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from kyos import lstm


def test_create_lstm_model():
    """Verify LSTM model creation with specified input shape and number of classes."""
    input_shape = (10, 20)
    num_classes = 5
    model = lstm.create_lstm_model(input_shape, num_classes)

    assert isinstance(model, Sequential)
    assert model.input_shape == (None, 10, 20)
    assert model.output_shape == (None, num_classes)


def test_create_lstm_model_with_different_neurons():
    """Verify LSTM model creation with different neuron configurations."""
    input_shape = (15, 25)
    num_classes = 3
    neurons = [50, 40, 30]
    model = lstm.create_lstm_model(input_shape, num_classes, neurons=neurons)

    assert isinstance(model, Sequential)
    assert model.input_shape == (None, 15, 25)
    assert model.output_shape == (None, num_classes)


def test_create_lstm_model_with_different_optimizer():
    """Verify LSTM model creation with different optimizers."""
    input_shape = (10, 20)
    num_classes = 5
    model = lstm.create_lstm_model(input_shape, num_classes, optimizer='Adam')

    assert isinstance(model, Sequential)
    assert model.input_shape == (None, 10, 20)
    assert model.output_shape == (None, num_classes)


if __name__ == '__main__':
    import pytest

    pytest.main()