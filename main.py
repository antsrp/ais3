import statistics

import numpy as np
import scipy.special
import imageio as ii

import viewer


class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def predict_by_image(self, image):
        array = np.array(image)
        data = 255.0 - array.reshape(784)
        data = normalize(np.asfarray(data))
        outputs = self.query(data)
        return np.argmax(outputs)

    def predict(self, filename):
        image = image_read(filename)
        outputs = self.query(image)
        return np.argmax(outputs)

    def saveWeights(self):
        np.save('trainingResults/wih.npy', self.wih)
        np.save('trainingResults/who.npy', self.who)

    def loadWeights(self, wihPath, whoPath):
        self.wih = np.load(wihPath)
        self.who = np.load(whoPath)


def normalize(array):
    return (array / 255.0 * 0.99) + 0.01


def image_read(filename):
    ir = ii.imread(filename)
    if len(ir.shape) > 2:
        array = ir.mean(axis=2)
    else:
        thresh = 127
        array = (ir > thresh) * 255
    data = array.reshape(784)

    mode = statistics.mode(data)
    if mode == 255:
        data = 255.0 - data

    data = normalize(np.asfarray(data))
    return data


def training(n, output_nodes, times):
    training_data_file = open("mnist_db/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    for _ in range(times):
        for record in training_data_list:
            all_values = record.split(',')

            inputs = normalize(np.asfarray(all_values[1:]))

            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

    return n


def testing(n):
    test_data_file = open("mnist_db/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    scores = []

    for test in test_data_list:
        all_values = test.split(',')
        image_array = np.asfarray(all_values[1:]).reshape((28, 28))

        inputs = normalize(np.asfarray(all_values[1:]))
        outputs = n.query(inputs)
        answer = np.argmax(outputs)

        if answer == int(all_values[0]):
            scores.append(1)
        else:
            scores.append(0)

    # print(scores)
    pass


def main():
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10

    learning_rate = 0.3

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # training(n, output_nodes, 5)
    # n.saveWeights()
    # testing(n)

    n.loadWeights('trainingResults/wih.npy', 'trainingResults/who.npy')

    layout = viewer.MainLayout(n)
    layout.mainLoop()


if __name__ == "__main__":
    main()
