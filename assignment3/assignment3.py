import _pickle as cPickle
from PIL import Image
import numpy as np
import random
import math
import gzip
import sys


def to_onehot_vector(dataset):
    actual_digits = []
    for digit in dataset[1]:
        onehot_v = [0] * 10
        onehot_v[digit] = 1
        actual_digits.append(onehot_v)
    return (dataset[0], np.array(actual_digits))


def get_chunks(last, n):
    for i in range(0, len(last), n):
        yield last[i:i + n]


def get_batches(train_set_pixel_lists, train_set_actual_digits, batch_size):
    return (np.array(list(get_chunks(train_set_pixel_lists, batch_size))),
            np.array(list(get_chunks(train_set_actual_digits, batch_size))))


def initialize_network(inputs, hidden, outputs):
    network = dict()
    network["hidden_layer"] = dict()
    network["hidden_layer"]["weights"] = np.random.normal(0, 1 / math.sqrt(inputs), size=(hidden, inputs))

    network["output_layer"] = dict()
    network["output_layer"]["weights"] = np.random.normal(0, 1 / math.sqrt(hidden), size=(outputs, hidden))

    network["hidden_layer"]["biases"] = np.random.normal(0, 1 / math.sqrt(inputs), hidden)
    network["output_layer"]["biases"] = np.random.normal(0, 1 / math.sqrt(outputs), outputs)

    return network


def sigmoid(list_to_activate):
    return 1 / (1 + np.exp(-list_to_activate))


def sigm_deriv(list_to_activate):
    return list_to_activate * (1.0 - list_to_activate)


def softmax(list_to_activate):
    return np.exp(list_to_activate) / sum(np.exp(list_to_activate))


def train(network, train_set_pixel_lists, train_set_actual_digits, learning_rate, num_of_epochs):
    batch_size = 10
    pixel_lists_batches, actual_digits_batches = get_batches(
        train_set_pixel_lists, train_set_actual_digits, batch_size)

    for epoch in range(num_of_epochs):
        matches = 0

        for i in range(len(pixel_lists_batches)):

            hidden_layer_partial_grad_weight = 0
            hidden_layer_partial_grad_bias = 0
            output_layer_partial_grad_w = 0
            output_layer_partial_grad_b = 0

            for index, pixel_list in enumerate(pixel_lists_batches[i]):
                t = actual_digits_batches[i][index]

                z = np.add(np.dot(network["hidden_layer"]["weights"], pixel_list), network["hidden_layer"]["biases"])
                network["hidden_layer"]["outputs"] = sigmoid(z)

                z = np.add(np.dot(network["output_layer"]["weights"], network["hidden_layer"]["outputs"]), network["output_layer"]["biases"])
                network["output_layer"]["outputs"] = softmax(z)

                if np.argmax(network["output_layer"]["outputs"]) == np.argmax(t):
                    matches += 1

                network["output_layer"]["error"] = t - network["output_layer"]["outputs"] # cross

                network["hidden_layer"]["error"] = np.multiply(sigm_deriv(network["hidden_layer"]["outputs"]),
                                                               np.dot(np.transpose(np.matrix(
                                                                   network["output_layer"]["weights"])), np.transpose(
                                                                   np.matrix(
                                                                       network["output_layer"]["error"]))).flatten())
                hidden_layer_partial_grad_weight += np.transpose(np.dot(np.transpose(np.matrix(pixel_list)), np.matrix(network["hidden_layer"]["error"])))
                hidden_layer_partial_grad_bias += network["hidden_layer"]["error"]

                output_layer_partial_grad_w += np.transpose(np.dot(np.transpose(np.matrix(network["hidden_layer"]["outputs"])), np.matrix(network["output_layer"]["error"])))
                output_layer_partial_grad_b += network["output_layer"]["error"]
            
            
            # invisibility = [0] * len(network["hidden_layer"]["biases"])
            # index_list = random.sample([i for i in range(len(invisibility))], int(len(invisibility)/2))
            # for index in index_list:
            #     invisibility[index] = 1
            # invisibility = np.array(invisibility)
            
            network["output_layer"]["weights"] += learning_rate / batch_size * output_layer_partial_grad_w
            # network["output_layer"]["weights"] += learning_rate / batch_size * (np.multiply(output_layer_partial_grad_w, np.matrix(np.broadcast_to(invisibility, (len(output_layer_partial_grad_w), len(invisibility))))))
            network["output_layer"]["biases"] += learning_rate / batch_size * output_layer_partial_grad_b

            network["hidden_layer"]["weights"] += learning_rate / batch_size * hidden_layer_partial_grad_weight
            # network["hidden_layer"]["weights"] += learning_rate / batch_size * np.multiply(np.broadcast_to(np.matrix(invisibility).T, (len(invisibility), len(pixel_list))), hidden_layer_partial_grad_weight)
            network["hidden_layer"]["biases"] += learning_rate / batch_size * np.squeeze(np.asarray(hidden_layer_partial_grad_bias))

        print("epoch " + str(epoch + 1) + ": " + str(100 * matches / len(train_set_pixel_lists)) + "%")
        # learning_rate = round(learning_rate * 0.9, 5)
    return network


def test(network, pixel_lists, actual_digits):
    dataset_size = len(pixel_lists)
    matches = 0
    for index, pixel_list in enumerate(pixel_lists):
        t = actual_digits[index]

        z = np.add(np.dot(network["hidden_layer"]["weights"], pixel_list), network["hidden_layer"]["biases"])
        network["hidden_layer"]["outputs"] = sigmoid(z)

        z = np.add(np.dot(network["output_layer"]["weights"], network["hidden_layer"]["outputs"]), network["output_layer"]["biases"])
        network["output_layer"]["outputs"] = softmax(z)

        if np.argmax(network["output_layer"]["outputs"]) == np.argmax(t):
            matches += 1
    print(str(100 * matches / dataset_size) + "%")


if __name__ == "__main__":
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
    f.close()

    train_set_pixel_lists, train_set_actual_digits = to_onehot_vector(train_set)
    validation_set_pixel_lists, validation_set_actual_digits = to_onehot_vector(valid_set)
    test_set_pixel_lists, test_set_actual_digits = to_onehot_vector(test_set)

    if "train" in sys.argv[1]:
        network_sizes = [784, 100, 10]
        network = initialize_network(network_sizes[0], network_sizes[1], network_sizes[2])
        trained_network = train(network, train_set_pixel_lists, train_set_actual_digits, 0.5, 30)
        with open('network.pkl', 'wb') as f:
            cPickle.dump(trained_network, f)
    elif "valid" in sys.argv[1]:
        with open('network.pkl', 'rb') as f:
            trained_network = cPickle.load(f)
        test(trained_network, validation_set_pixel_lists, validation_set_actual_digits)
    elif "test" in sys.argv[1]:
        with open('network.pkl', 'rb') as f:
            trained_network = cPickle.load(f)
        test(trained_network, test_set_pixel_lists, test_set_actual_digits)
