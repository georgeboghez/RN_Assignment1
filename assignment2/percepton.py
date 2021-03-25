import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image

class Perceptron:
    def __init__(self, label, weight, bias, learningRate, accuracy = 0, falsePositives = 0, falseNegatives = 0):
        self.__label = label
        self.__weight = np.array(weight)
        self.__bias = bias
        self.__learningRate = learningRate
        self.__accuracy = accuracy
        self.__falsePositives = falsePositives
        self.__falseNegatives = falseNegatives
    
    def activate(self, input):
        if input > 0:
            return 1
        return 0
    
    def chunks(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    def mini_batch(self, train_set, iterations):
        pixelsBatches = list(self.chunks(train_set[0], 5000))
        numsBatches = list(self.chunks(train_set[1], 5000))
        # print(len(pixelsBatches))

        while iterations > 0:
            num = 0
            falseNegatives = 0
            falsePositives = 0
            for i in range(len(pixelsBatches)):
                delta_i = 0
                beta_i = 0
                for j in range(len(pixelsBatches[i])):
                    if numsBatches[i][j] == self.__label:
                        t = 1
                    else:
                        t = 0
                    z = np.dot(self.__weight, pixelsBatches[i][j]) + self.__bias
                    output = self.activate(z)
                    delta_i = delta_i + (t - output) * pixelsBatches[i][j] * self.__learningRate
                    beta_i = beta_i + (t - output) * self.__learningRate
                    if output != t:
                        num += 1
                        if output == 0 and t == 1:
                            falseNegatives += 1
                        else:
                            falsePositives += 1
                self.__weight = self.__weight + delta_i
                self.__bias = self.__bias + beta_i
            accuracy = 100 - (100 * num) / 50000
            print(iterations, accuracy)
            iterations -= 1
        self.__accuracy = accuracy
        self.__falsePositives = falsePositives
        self.__falseNegatives = falseNegatives
        return accuracy
    
    def activateMatrix(self, input):
        return [1 if input[i] > 0 else 0 for i in range(len(input))]

    def mini_batch_matrix(self, train_set, iterations):
        pixelsBatches = list(self.chunks(train_set[0], 5000))
        numsBatches = list(self.chunks(train_set[1], 5000))
        # print(len(pixelsBatches))
        
        weightMatrixNP = []
        for row in pixelsBatches[0]:
            weightMatrixNP.append(np.array(self.__weight))
        
        weightMatrixNP = np.array(weightMatrixNP)
        
        for i in range(len(pixelsBatches)):
            t = [1 if numsBatches[i][j] == self.__label else 0 for j in range(len(numsBatches[i]))]
            z = np.dot(weightMatrixNP.T, pixelsBatches[i]) + self.__bias
            output = self.activateMatrix(z)
            # delta_i = delta_i

    
    def train_data(self, train_set, iterations):
        allClassified = False
        accuracy = 0
        while not allClassified and iterations > 0:
            allClassified = True
            num = 0
            falseNegatives = 0
            falsePositives = 0
            for index in range(len(train_set[0])):
                if train_set[1][index] == self.__label:
                    t = 1
                else:
                    t = 0
                z = np.dot(self.__weight, train_set[0][index]) + self.__bias
                output = self.activate(z)
                self.__weight = self.__weight + (t - output) * train_set[0][index] * self.__learningRate
                self.__bias = self.__bias + (t - output) * self.__learningRate
                if output != t:
                    num += 1
                    allClassified = False
                    if output == 0 and t == 1:
                        falseNegatives += 1
                    else:
                        falsePositives += 1

            accuracy = 100 - (num* 100)/ 50000
            print(iterations, accuracy)

            iterations -= 1
        # a = np.amin(self.__weight)
        # c = np.amax(self.__weight)
        # y=(self.__weight-a)/(c-a)*(0-1)+1

        # self.convert_to_image(y, self.__label, iterations, self.__weight[0], self.__bias, self.__learningRate)
        self.__accuracy = accuracy
        self.__falsePositives = falsePositives
        self.__falseNegatives = falseNegatives
        return accuracy
    
    def train_matrix(self, train_set, iterations):
        # print(len(pixelsBatches))
        
        weightMatrixNP = []
        self.__bias = [self.__bias for i in range(len(train_set[0]))]
        print(len(self.__bias))
        for row in range(len(train_set[0])):
            weightMatrixNP.append(np.array(self.__weight))
        
        weightMatrixNP = np.array(weightMatrixNP)
        print("a", len(weightMatrixNP), len(weightMatrixNP[0]))
    
        # t = [1 if train_set[i][j] == self.__label else 0 for j in range(len(numsBatches[i]))]
        # z = np.dot(weightMatrixNP.T, pixelsBatches[i]) + self.__bias
        # output = self.activateMatrix(z)
        # weightMatrixNP = weightMatrixNP + (t - output) * pixelsBatches[i] * self.__learningRate
        # self.__bias = self.__bias + (t - output) * self.__learningRate
    
    def check_result(self, digitPixelList):
        return np.dot(self.__weight, digitPixelList) + self.__bias
    
    def calc_res(self, digitPixelList):
        return np.dot(self.__weight, digitPixelList) + self.__bias
    
    def get_label(self):
        return self.__label
    
    def set_label(self, label):
        self.__label = label
        
    def get_weight(self):
        return self.__weight

    def set_weight(self, weight):
        self.__weight = weight

    def get_bias(self):
        return self.__bias

    def set_bias(self, bias):
        self.__bias = bias

    def get_learning_rate(self):
        return self.__learningRate

    def set_learning_rate(self, learning_rate):
        self.__learningRate = learning_rate
    
    def get_accuracy(self):
        return self.__accuracy
    
    def set_accuracy(self, acc):
        self.__accuracy = acc
    
    def toJSON(self):
        return json.dumps({"weight": self.__weight.tolist(), "bias": self.__bias, "learning_rate": self.__learningRate, "accuracy": self.__accuracy, "false_positives": self.__falsePositives, "false_negatives": self.__falseNegatives, "label": self.__label}, indent=4)

    def convert_to_image(self, arr, label, i, w, b, l):
        slice56 = arr.reshape(28,28)
        formatted = (slice56 * 255 / np.max(slice56)).astype('uint8')
        img = Image.fromarray(formatted)
        img.save("./imgs_/img" + str(label) + "w" + str(w) + "b" + str(b) + "l" + str(l) +  "it" + str(i) + ".png")
        # img.show()
