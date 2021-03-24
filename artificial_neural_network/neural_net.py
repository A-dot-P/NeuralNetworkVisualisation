import os

import numpy as np
from scipy import ndimage


class Network:
    def __init__(self, nodes_per_layer: list, learning_rate: float = 0.1, load_folder=None):
        """
        Creates network of specified dimensions, either imports weights from folder specified by load folder,
        or randomly generates weights between 0 and 1 and sets biases to 0
        :param nodes_per_layer: a list with an integer number for the number of nodes on a given layer
        :param learning_rate: measure of how much the weights change each training round
        :param load_folder: location to find a particular trained network's folder in the trained_networks
                            subdirectory from which existing weight values can be imported
        """
        self.nodes_per_layer = nodes_per_layer
        self.learning_rate = learning_rate
        self.network_folder = "trained_networks"
        if load_folder:
            self.load(load_folder)
        else:
            self.weights = [np.random.normal(0.0, 1, (current_layer_nodes, next_layer_nodes))
                            for current_layer_nodes, next_layer_nodes in zip(nodes_per_layer, nodes_per_layer[1:])]
            self.biases = [np.zeros(layer) for layer in nodes_per_layer[1:]]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.nodes_per_layer!r})"

    @staticmethod
    def solenoid(x):
        return 1 / (1 + np.exp(x) ** -1)

    @staticmethod
    def feed_forward(data: np.array, weights: np.array, bias: np.array):
        """
        Determine activation of next layer of nodes given the previous layer's activation
        :param data: The previous layer of nodes' activations
        :param weights: The weights from the previous layer of nodes to the current nodes
        :param bias: The current layer of node's biases
        :return: The current layer of nodes' activations
        """
        return Network.solenoid(np.dot(data, weights) + bias)

    def query(self, input_data: np.array) -> np.array:
        """
        Run feed_forward on all layers to determine activation of all nodes for a given input
        :param input_data: array with same dimensions as first layer of nodes
        :return: array with the activations of nodes in each layer
        """
        hidden_data = list(input_data[np.newaxis])
        for layer in range(len(self.nodes_per_layer) - 1):
            hidden_data.append(self.feed_forward(hidden_data[-1], self.weights[layer], self.biases[layer]))
        return hidden_data

    def query_outputs(self, input_data: np.array) -> np.array:
        """
        Run feed_forward on all layers to determine activation of all nodes for a given input
        :param input_data: array with same dimensions as first layer of nodes
        :return: the last layer of nodes' activation
        """
        return self.query(input_data)[-1]

    def train(self, input_data: np.array, target: np.array):
        """
        trains the network for one given input and the expected output activation
        :param input_data: array with same dimensions as first layer of nodes
        :param target: array with same dimensions as last layer of nodes
        """
        query_data = self.query(input_data)
        derivative_loss_func = target - query_data[-1]  # activations[-1 (L)] - y [* sigmoid prime of L(-1)]
        for layer in range(len(self.nodes_per_layer) - 2, 0, -1):  # update weight layers 0 to L-1
            layer_input = query_data[layer]  # layer l-1's activations
            layer_output = query_data[layer + 1]  # sigmoid(z) # l's activations
            derivative_activation_func = layer_output * (1 - layer_output)  # sigmoid prime

            delta = derivative_loss_func * derivative_activation_func
            self.weights[layer] += self.learning_rate * np.outer(layer_input, delta)
            self.biases[layer] += self.learning_rate * delta
            derivative_loss_func = np.dot(self.weights[layer], delta)

    def save(self, folder_name):
        """
        Save a trained network's weights and biases to a folder in the sub-directory network_folder for reuse
        """
        try:
            os.mkdir(f"{self.network_folder}/{folder_name}")
        except FileExistsError:
            print(f"Warning - over-writing folder {folder_name}")
        np.save(f"{self.network_folder}/{folder_name}/weights.npy", np.asarray(self.weights))
        np.save(f"{self.network_folder}/{folder_name}/biases.npy", np.asarray(self.biases))

    def load(self, folder_name):
        """
        Load a trained network's weights and biases from a folder in the sub-directory network_folder
        """
        self.weights = np.load(f"{self.network_folder}/{folder_name}/weights.npy", allow_pickle=True)
        self.biases = np.load(f"{self.network_folder}/{folder_name}/biases.npy", allow_pickle=True)

    @staticmethod
    def get_data(filename):
        "Convert black and white images with integer greyscale values to floats"
        data_set = np.loadtxt(filename, delimiter=',', dtype=float)
        data_set[:, 1:] = data_set[:, 1:] / 255 * 0.99 + 0.001
        return data_set

    def accuracy_test(self, data_filename):
        data = self.get_data(data_filename)
        correct = 0
        print(f"Target Prediction Activation Result\n{'-' * 36}")
        for record in data:
            target, x, = np.split(record, [1])
            prediction = self.query_outputs(x)
            print(f"{str(int(target.item())):^6} {str(np.argmax(prediction)):^10}  [{np.max(prediction):.2f}]   " +
                  ('CORRECT' if np.argmax(prediction) == target else 'INCORRECT'))
            if np.argmax(prediction) == target:
                correct += 1
        print(f"Accuracy: {correct / np.size(data, axis=0):.1%}")

    def train_from_data_set(self, data_filename, epochs=10, save_folder_name=None, angles=(5, 10)):
        """
        Train network in all files
        :param data_filename: Training data
        :param epochs: Number of times each sample is used
        :param save_folder_name: If not None, then trained network can immediately be saved to a folder
        :param angles: List of positive angles that training will be rotated clockwise an anticlockwise by
                        and the network also trained with these inputs (the angle 0 does not need to be specified)
        :return:
        """
        data = self.get_data(data_filename)
        for epoch in range(epochs):
            np.random.shuffle(data)
            for i, record in enumerate(data):
                y, x, = np.split(record, [1])
                targets = np.where(np.equal(int(y), np.arange(self.nodes_per_layer[-1])), 0.99, 0.01)
                self.train(x, targets)
                for angle in angles:
                    clockwise_rotated_x = ndimage.interpolation.rotate(x.reshape(28, 28), angle, cval=0.01, order=1,
                                                                       reshape=False)
                    anticlockwise_rotated_x = ndimage.interpolation.rotate(x.reshape(28, 28), -angle, cval=0.01,
                                                                           order=1,
                                                                           reshape=False)
                    self.train(clockwise_rotated_x.flatten(), targets)
                    self.train(anticlockwise_rotated_x.flatten(), targets)

                print('\r' + f"{'=' * epoch}{'-' * (epochs - epoch)} {(i + 1) / np.size(data, axis=0):.0%}", end='')
        print('\n')
        if save_folder_name:
            self.save(save_folder_name)


if __name__ == '__main__':
    np.random.seed(10)
    np.set_printoptions(precision=3, threshold=10, edgeitems=3, suppress=True)

    training_network = Network([784, 75, 10])
    training_network.train_from_data_set("data_sets/mnist_train.csv")
    training_network.accuracy_test("data_sets/mnist_test.csv")
