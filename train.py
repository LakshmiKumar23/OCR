import sys
import numpy as np
from perceptron import train_multiclass_perceptron, train_binary_perceptron
from pre_process import readfile, get_features_for_digits, getlabels, getsamples, get_features_for_faces


def train_data(train_data, train_labels, type_of_data, algorithm):
    if type_of_data == 1:
        height = 28
        width = 28
        classes = 10
    else:
        height = 70
        width = 60
        classes = 2
    samples, sample_lines = readfile(train_data, type_of_data)
    samples = getsamples(samples, sample_lines, height, width)
    labels, label_lines = readfile(train_labels, type_of_data)
    labels = getlabels(labels)
    if type_of_data == 1:
        feature_matrix = get_features_for_digits(samples)
        if algorithm == 'perceptron':
            weights_perceptron = train_multiclass_perceptron(feature_matrix, labels, classes)
            np.save('digits_perceptron_weights', weights_perceptron)
        if algorithm == 'knn':
            np.save('digits_knn_features', feature_matrix)
    else:
        feature_matrix = get_features_for_faces(samples)
        if algorithm == 'perceptron':
            weights_perceptron = train_binary_perceptron(feature_matrix, labels)
            np.save('faces_perceptron_weights', weights_perceptron)
        if algorithm == 'knn':
            np.save('faces_knn_features', feature_matrix)


if __name__ == '__main__':
    train_data(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
