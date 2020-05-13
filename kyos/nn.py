# -*- coding: utf-8 -*-

"""This module is the neural network portion of Kyos.
"""

from __future__ import print_function
from __future__ import absolute_import

import logging
import pandas as pd
import psutil
import random
import numpy as np
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from kyos import features
from kyos.__init__ import __version__


# Zero-based index to first feature column.
first_ftr_idx = features.feature_names.index(features.first_ftr_name)

# Zero-based index to last feature column.
last_ftr_idx = features.feature_names.index(features.last_ftr_name)

num_input_features = 1 + last_ftr_idx - first_ftr_idx

# the truth column is appended to feature_names when the truth file is provided
target_label_name = features.target_label_name

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


def conv_allele(allele):
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
        raise ValueError("Unknown allele: %s" % (allele))


def conv_output(output):
    for x in range(len(output)):
        if output[x] > 0.7:
            return output_classes[x]

    return "-"


def normalize_features(df):
    """Normalize the input features so the scaled features are scaled approximately in the range 0 to 1.

    In practice, we found better results when the read count features are scaled in such a way that some
    of the values are greater than 1.

    The datafame is modified in-place.

    Parameters
    ----------
    df : Pandas dataframe
        Dataframe of features
    """
    # The scaling parameters below are hard-coded to be sure the same values will be used for training, testing,
    # and calling variants.  It might happen that new datasets will have different distribution of values.
    for ftr_name in features.read_count_feature_names:
        df[ftr_name] = df[ftr_name] / 20.0  # 60th percentile, so some values will scale higher than 1
    for ftr_name in features.map_quality_feature_names:
        df[ftr_name] = df[ftr_name] / 100.0
    for ftr_name in features.base_quality_feature_names:
        df[ftr_name] = df[ftr_name] / 100.0


def standardize_features(df):
    """Standardize the input features so the scaled features are approximately centered around 0 with variance 1.

    The datafame is modified in-place.

    Parameters
    ----------
    df : Pandas dataframe
        Dataframe of features
    """
    # The scaling parameters below are hard-coded to be sure the same values will be used for training, testing,
    # and calling variants.  It might happen that new datasets will have different distribution of values.
    for ftr_name in features.read_count_feature_names:
        df[ftr_name] = (df[ftr_name] - 6.0) / 11.0
    for ftr_name in features.map_quality_feature_names:
        df[ftr_name] = (df[ftr_name] - 50.0) / 50.0
    for ftr_name in features.base_quality_feature_names:
        df[ftr_name] = (df[ftr_name] - 50.0) / 50.0


def load_train_data(path, first_ftr_col, last_ftr_col, scaling="normalize"):
    """Load training file with features and target class.

    The features must be adjacent columns.
    The target must be the last column.
    It is okay if there are some unused columns before the first feature and after the last feature.

    Parameters
    ----------
    path : str
        Path to tabulated feature file.
    first_ftr_col : int
        Zero-based index to first feature column.
    first_ftr_col : int
        Zero-based index to last feature column.
    scaling : str, optional
        String specifying how to scale the input features.
        Possible values are: "normalize" and "standardize".  By default, features are normalized.

    Returns
    -------
    features : nparray
        Two dimensional array of features with one row per observation and one column per feature
    one_hot_labels : nparray
        One hot encoded target labels with one row per observation and one column per output class.
        All columns will be 0 except the target column will be 1.
    """
    start = timer()

    with open(path, "r") as csv_file:
        logging.debug("Reading tsv file...")
        feature_columns = features.feature_names[first_ftr_col: 1 + last_ftr_col]
        usecols = feature_columns + [target_label_name]
        dtype = {col: np.float32 for col in feature_columns}
        converters = {target_label_name: conv_allele}
        df = pd.read_csv(csv_file, sep='\t', usecols=usecols, dtype=dtype, converters=converters)

    if scaling == "normalize":
        logging.debug("Normalizing features...")
        normalize_features(df)
    elif scaling == "standardize":
        logging.debug("Standardizing features...")
        standardize_features(df)

    logging.debug("Extracting columns to np arrays...")
    data = df[feature_columns].values
    labels = df[target_label_name].values

    logging.debug("One-hot encoding target labels...")
    one_hot_labels = keras.utils.to_categorical(labels, num_classes=9)

    end = timer()
    logging.debug("%.1f seconds loading %d rows in file %s" % (end - start, len(df), path))
    return data, one_hot_labels


def load_test_data(path, first_ftr_col, last_ftr_col, scaling="normalize"):
    """Load testing file with features and target class.

    The features must be adjacent columns.
    The target must be the last column.
    It is okay if there are some unused columns before the first feature and after the last feature.

    Parameters
    ----------
    path : str
        Path to tabulated feature file.
    first_ftr_col : int
        Zero-based index to first feature column.
    first_ftr_col : int
        Zero-based index to last feature column.
    scaling : str, optional
        String specifying how to scale the input features.
        Possible values are: "normalize" and "standardize".  By default, features are normalized.

    Returns
    -------
    features : nparray
        Two dimensional array of features with one row per observation and one column per feature
    labels : nparray
        Integer encoded labels with one row per observation
    refs : nparray
        Integer encoded references with one row per observation
    """
    start = timer()

    with open(path, "r") as csv_file:
        logging.debug("Reading tsv file...")
        feature_columns = features.feature_names[first_ftr_col: 1 + last_ftr_col]
        usecols = feature_columns + [target_label_name] + ["RefBase"]
        dtype = {col: np.float32 for col in feature_columns}
        converters = {target_label_name: conv_allele, "RefBase": conv_allele}
        df = pd.read_csv(csv_file, sep='\t', usecols=usecols, dtype=dtype, converters=converters)

    if scaling == "normalize":
        logging.debug("Normalizing features...")
        normalize_features(df)
    elif scaling == "standardize":
        logging.debug("Standardizing features...")
        standardize_features(df)

    logging.debug("Extracting columns to np arrays...")
    data = df[feature_columns].values
    labels = df[target_label_name].values
    refs = df["RefBase"].values

    end = timer()
    logging.debug("%.1f seconds loading %d rows in file %s" % (end - start, len(df), path))
    return data, labels, refs


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
        cores = 1
    else:
        logging.info("************************************************************************************************")
        logging.info("NOTICE: results are not reproducible when rseed is not set.")
        logging.info("************************************************************************************************")

        # Use all CPUs
        cores = psutil.cpu_count(logical=True)

    logging.info("Using %d CPUs", cores)
    tf.config.threading.set_inter_op_parallelism_threads(cores)
    tf.config.threading.set_intra_op_parallelism_threads(cores)

    logging.debug("Kyos train, version %s" % __version__)
    logging.debug("Loading data...")
    data, one_hot_labels = load_train_data(train_file_path, first_ftr_idx, last_ftr_idx)
    data_validation, one_hot_label_validation = load_train_data(validate_file_path, first_ftr_idx, last_ftr_idx)

    logging.debug("Defining model...")
    model = Sequential()
    model.add(Dense(40, input_dim=num_input_features))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(30))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(30))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(30))
    model.add(Activation("relu"))

    model.add(Dense(9, activation='softmax'))

    optimizer = keras.optimizers.RMSprop(lr=0.0005)

    logging.debug("Compiling model...")
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping_monitor = EarlyStopping(patience=10, restore_best_weights=True)
    callbacks = [early_stopping_monitor]

    logging.debug("Fitting model...")
    model.fit(data, one_hot_labels, validation_data=(data_validation, one_hot_label_validation), batch_size=100000, callbacks=callbacks, epochs=100, verbose=2)

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

    data, labels, reference = load_test_data(test_file_path, first_ftr_idx, last_ftr_idx)

    model = keras.models.load_model(model_file_path)

    prediction = model.predict(data)

    output = np.argmax(prediction, axis=1)

    df = pd.DataFrame({"output": output, "labels": labels, "reference": reference})

    df["Mutation"] = df.labels != df.reference

    df["Correct_call"] = df.output == df.labels
    df["TN"] = (~df.Mutation) & df.Correct_call
    df["TP"] = df.Mutation & df.Correct_call
    df["FP"] = (~df.Mutation) & (~df.Correct_call)
    df["FN"] = df.Mutation & (~df.Correct_call)

    false_positives = df["FP"].sum()
    false_negatives = df["FN"].sum()
    true_positives = df["TP"].sum()
    true_negatives = df["TN"].sum()

    print("FP:", false_positives, "FN:", false_negatives, "TP:", true_positives, "TN:", true_negatives)


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
    print('The "call" command is not implemented yet')
