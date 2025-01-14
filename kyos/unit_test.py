# -*- coding: utf-8 -*-

import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from kyos.lstm import create_lstm_model
from kyos.nn import load_train_data, load_test_data, train, test

def test_create_lstm_model():
    input_shape = (10, 20)
    num_classes = 5
    model = create_lstm_model(input_shape, num_classes)
    assert isinstance(model, Sequential)
    assert model.input_shape == (None, 10, 20)
    assert model.output_shape == (None, num_classes)

def test_load_train_data():
    path = 'tests/data/train_data.tsv'
    first_ftr_col = 0
    last_ftr_col = 10
    data, one_hot_labels = load_train_data(path, first_ftr_col, last_ftr_col)
    assert data.shape[1] == last_ftr_col - first_ftr_col + 1
    assert one_hot_labels.shape[1] == 9  # Number of output classes

def test_load_test_data():
    path = 'tests/data/test_data.tsv'
    first_ftr_col = 0
    last_ftr_col = 10
    data, labels, refs = load_test_data(path, first_ftr_col, last_ftr_col)
    assert data.shape[1] == last_ftr_col - first_ftr_col + 1
    assert labels.shape[0] == data.shape[0]
    assert refs.shape[0] == data.shape[0]

def test_train():
    train_file_path = 'tests/data/train_data.tsv'
    validate_file_path = 'tests/data/validate_data.tsv'
    model_file_path = 'tests/models/test_model.h5'
    train(train_file_path, validate_file_path, model_file_path, rseed=42)
    model = tf.keras.models.load_model(model_file_path)
    assert isinstance(model, Sequential)

def test_test():
    model_file_path = 'tests/models/test_model.h5'
    test_file_path = 'tests/data/test_data.tsv'
    test(model_file_path, test_file_path)
    # Add assertions based on expected output

if __name__ == '__main__':
    pytest.main()