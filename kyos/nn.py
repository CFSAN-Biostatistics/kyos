from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
import numpy as np
import csv

features = ["A", "T", "C", "G", "A_insertion", "T_insertion", "C_insertion", "G_insertion", "deletion"]


def relevant_data(data):
    z = []
    for x in range(len(data)):
        if x > 2 and x < 29:
            z.append(float(data[x]))
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
        if allele[1].upper() == "A":
            return 4
        elif allele[1].upper() == "T":
            return 5
        elif allele[1].upper() == "C":
            return 6
        elif allele[1].upper() == "G":
            return 7
    elif "_deletion" in allele:
        return 8
    else:
        print("Unkown Allele " + allele)
        print(row)
        return -1


def conv_output(output):
    for x in range(len(output)):
        if output[x] > 0.7:
            return features[x]

    return "-"


def train(train_file_path, validate_file_path, model_file_path):
    """Train a neural network to detect variants.

    Parameters
    ----------
    train_file_path : str
        Input tabulated feature file for training.
    validate_file_path : str
        Input tabulated feature file for validation during training.
    model_file_path : str
        Output trained model.
    """

    data = []
    labels = []

    line_count = 0

    with open(train_file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            if line_count == 0:
                print("Column names are, " + str(row))
                line_count += 1
            else:
                data.append(relevant_data(row))
                labels.append(conv_allele(row[-1]))

    data_validation = []
    label_validation = []

    line_count = 0

    with open(validate_file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            if line_count == 0:
                print("Column names are, " + str(row))
                line_count += 1
            else:
                data_validation.append(relevant_data(row))
                label_validation.append(conv_allele(row[-1]))

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
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    data = np.array(data)
    data_validation = np.array(data_validation)

    early_stopping_monitor = EarlyStopping(patience=3)

    one_hot_labels = keras.utils.to_categorical(labels, num_classes=9)

    one_hot_label_validation = keras.utils.to_categorical(label_validation, num_classes=9)

    model.fit(data, one_hot_labels, validation_data=(data_validation, one_hot_label_validation), batch_size=128, callbacks=[early_stopping_monitor], epochs=30)

    model.save(model_file_path)


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
                data.append(relevant_data(row))
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

        truth_call = features[labels[row]]

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
