# -*- coding: utf-8 -*-

"""This module is the neural network portion of Kyos.
"""

from __future__ import print_function
from __future__ import absolute_import

import csv
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
import logging
import random
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer

from kyos import features

output_classes = ["A", "T", "C", "G", "A_insertion", "T_insertion", "C_insertion", "G_insertion", "_deletion"]


def relevant_data(data, first_col, last_col):
    """Extract the specified range of adjacent feature columns from a single observation row.

    Parameters
    ----------
    data : list
        List of feature and target columns.
    first_col : int
        First column index to select.
    last_col : int
        Last column index to select.

    Examples
    --------
    >>> relevant_data(list(range(0, 30)), 3, 28)
    [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0]
    """
    z = [float(val) for val in data[first_col: 1 + last_col]]
    return z


def conv_allele(row):
    allele = row[-1]

    if allele.upper() == "A":
        return 0
    elif allele.upper() == "T":
        return 1
    elif allele.upper() == "C":
        return 2
    elif allele.upper() == "G":
        return 3
    elif "_insertion" in allele:
        if allele[0].upper() == "A":
            return 4
        elif allele[0].upper() == "T":
            return 5
        elif allele[0].upper() == "C":
            return 6
        elif allele[0].upper() == "G":
            return 7
    elif "_deletion" in allele:
        return 8
    else:
        raise ValueError("Unkown allele: %s\nRow: %s" % (allele, row))


def conv_output(output):
    for x in range(len(output)):
        if output[x] > 0.7:
            return output_classes[x]

    return "-"


def load_train_data(path, first_ftr_col, last_ftr_col):
    """Load training file with features and target class.

    The features must be adjacent columns.
    The traget must be the last column.
    It is okay if there are some unused columns before the first feature and after the last feature.

    Parameters
    ----------
    path : str
        Path to tabulated feature file.
    first_ftr_col : int
        Zero-based index to first feature column.
    first_ftr_col : int
        Zero-based index to last feature column.

    Returns
    -------
    features : nparray
        Two dimensional array of features with one row per observation and one column per feature
    one_hot_labels : nparray
        One hot encoded target labels with one row per observation and one column per output class.
        All columns will be 0 except the target column will be 1.
    """
    start = timer()

    data = []
    labels = []
    row_count = 0

    with open(path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        csv_reader.next()  # skip header row
        # logging.debug("Column names are: %s" % str(header_row))
        for row in csv_reader:
            row_count += 1
            data.append(relevant_data(row, first_ftr_col, last_ftr_col))
            labels.append(conv_allele(row))

    logging.debug("Converting features to np arrays...")
    data = np.array(data, dtype=np.float32)

    logging.debug("One-hot encoding target labels...")
    one_hot_labels = keras.utils.to_categorical(labels, num_classes=9)

    end = timer()
    logging.debug("%.1f seconds loading %d rows in file %s" % (end - start, row_count, path))
    return data, one_hot_labels


def train(train_file_path, validate_file_path, model_file_path, rseed=None):
    """Train a neural network to detect variants.

    Parameters
    ----------
    train_file_path : str
        Input tabulated feature file for training.
    validate_file_path : str
        Input tabulated feature file for validation during training.
    model_file_path : str
        Output trained model.
    rseed : int
        Random seed to ensure reproducible results.  Set to zero for non-deterministic results.
    """
    if rseed:
        logging.info("************************************************************************************************")
        logging.info("NOTICE: setting the random seed also forces single-threaded execution to ensure reproducibility.")
        logging.info("************************************************************************************************")
        logging.debug("Setting random seed = %d" % rseed)
        random.seed(rseed)
        np.random.seed(rseed)
        tf.set_random_seed(rseed)

        # Limit operation to 1 thread for deterministic results.
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        keras.backend.set_session(sess)
    else:
        logging.info("************************************************************************************************")
        logging.info("NOTICE: results are not reproducible when rseed is not set.")
        logging.info("************************************************************************************************")

    logging.debug("Loading data...")
    data, one_hot_labels = load_train_data(train_file_path, features.first_ftr_idx, features.last_ftr_idx)
    data_validation, one_hot_label_validation = load_train_data(validate_file_path, features.first_ftr_idx, features.last_ftr_idx)

    logging.debug("Defining model...")
    model = Sequential()
    model.add(Dense(40, input_dim=26))
    model.add(Activation("relu"))

    model.add(Dense(30))
    model.add(Activation("relu"))

    model.add(Dense(30))
    model.add(Activation("relu"))

    model.add(Dense(30))
    model.add(Activation("relu"))

    model.add(Dense(9, activation='softmax'))

    logging.debug("Compiling model...")
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping_monitor = EarlyStopping(patience=3)

    logging.debug("Fitting model...")
    model.fit(data, one_hot_labels, validation_data=(data_validation, one_hot_label_validation), batch_size=128, callbacks=[early_stopping_monitor], epochs=30)

    logging.debug("Saving model...")
    model.save(model_file_path)
    logging.debug("Training finished.")


def test(model_file_path, test_file_path, vcf_file_path=None):
    """Test a neural network variant detector.

    Parameters
    ----------
    model_file_path : str
        Input trained model.
    test_file_path : str
        Input tabulated feature file for testing.
    vcf_file_path : str, optional
        Optional output VCF file.
    """
    data = []
    labels = []

    line_count = 0

    reference = []

    with open(test_file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            if line_count == 0:
                print("Column names are, " + str(row))
                line_count += 1
            else:
                data.append(relevant_data(row, features.first_ftr_idx, features.last_ftr_idx))
                reference.append(row[-2])
                labels.append(conv_allele(row))

    model = keras.models.load_model(model_file_path)

    false_positives = 0
    false_negatives = 0
    true_positive = 0
    true_negative = 0

    data = np.array(data)

    prediction = model.predict(data)

    for row in range(len(prediction)):

        predicted_output = conv_output(prediction[row])

        truth_call = output_classes[labels[row]]

        correct_call_flag = predicted_output == truth_call

        if correct_call_flag:
            if truth_call != reference[row]:
                true_positive += 1
            else:
                true_negative += 1
        else:  # incorrect call
            if truth_call == "-":
                continue
            if truth_call != reference[row]:
                false_negatives += 1
            else:
                false_positives += 1

    print("FP:", false_positives, "FN:", false_negatives, "TP:", true_positive, "TN:", true_negative)


def call(model_file_path, ftr_file_path, vcf_file_path):
    """Call variants from a tabulated feature file when the truth is unknown.

    Parameters
    ----------
    model_file_path : str
        Input trained model.
    ftr_file_path : str
        Input tabulated feature file.
    vcf_file_path : str, optional
        Optional output VCF file.
    """
    pass  # TODO
