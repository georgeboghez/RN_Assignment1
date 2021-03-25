import pickle, gzip
import random
import json
import copy
import sys
from percepton import Perceptron

def train(train_set):
    d = []
    for j in range(10):
        bestAccuracy = 0
        bestPerceptron = None
        for i in range(20):
            print(j, i)
            p = Perceptron(j, [random.uniform(-0.5, 0.5)] * len(train_set[0][0]), random.uniform(-0.5, 0.5), 0.001)
            accuracy = p.train_data(train_set, 30)
            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                bestPerceptron = copy.deepcopy(p)
        d.append(bestPerceptron.toJSON())
    with open('perceptrons.json', 'w+') as outfile:
        json.dump(d, outfile)

def train_mini(train_set):
    d = []
    for j in range(10):
        bestAccuracy = 0
        bestPerceptron = None
        for i in range(3):
            print(j, i)
            p = Perceptron(j, [random.uniform(-0.5, 0.5)] * len(train_set[0][0]), random.uniform(-0.5, 0.5), 0.001)
            accuracy = p.mini_batch(train_set, 30)
            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                bestPerceptron = copy.deepcopy(p)
        d.append(bestPerceptron.toJSON())
    with open('perceptrons_mini_batch.json', 'w+') as outfile:
        json.dump(d, outfile)

def check_set(_set):
    with open("perceptrons.json", "r") as f:
        perceptronList = json.loads(f.read())
        for i in range(len(perceptronList)):
            perceptronList[i] = json.loads(perceptronList[i])
            p = Perceptron(perceptronList[i]["label"], perceptronList[i]["weight"], perceptronList[i]["bias"], perceptronList[i]["learning_rate"], perceptronList[i]["accuracy"])
            perceptronList[i] = copy.deepcopy(p)
        
        num = 0
        for index, digit in enumerate(_set[0]):
            bestResult = -10000
            for perceptron in perceptronList:
                result = perceptron.check_result(digit)
                if result > bestResult:
                    bestPerceptron = perceptron.get_label()
            if bestPerceptron == _set[1][index]:
                num += 1
        print(100 * num / len(_set[1]))

def check_set_mini(_set):
    with open("perceptrons_mini_batch.json", "r") as f:
        perceptronList = json.loads(f.read())
        for i in range(len(perceptronList)):
            perceptronList[i] = json.loads(perceptronList[i])
            p = Perceptron(perceptronList[i]["label"], perceptronList[i]["weight"], perceptronList[i]["bias"], perceptronList[i]["learning_rate"], perceptronList[i]["accuracy"])
            perceptronList[i] = copy.deepcopy(p)
        
        num = 0
        for index, digit in enumerate(_set[0]):
            bestRes = -1000000
            for perceptron in perceptronList:
                result = perceptron.check_result(digit)
                if result > bestRes:
                    bestPerceptron = perceptron.get_label()
            if bestPerceptron == _set[1][index]:
                num += 1
        print(100 * num / len(_set[1]))



if __name__ == "__main__":
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin')
    f.close()

    if len(sys.argv) == 2:
        if sys.argv[1] == "online":
            train(train_set)
        elif sys.argv[1] == "mini":
            train_mini(train_set)
        elif sys.argv[1] == "miniM":
            p = Perceptron(0, [random.uniform(-0.5, 0.5)] * len(train_set[0][0]), random.uniform(-0.5, 0.5), 0.001)
            p.train_matrix(train_set, 1)
        elif sys.argv[1] == "validOnline":
            check_set(valid_set)
        elif sys.argv[1] == "testOnline":
            check_set(test_set)
        elif sys.argv[1] == "validMini":
            check_set_mini(valid_set)
        elif sys.argv[1] == "testMini":
            check_set_mini(test_set)
        else:
            print("invalid argument. try running the script with one of the following arguments: online, mini, validOnline, testOnline, validMini, testMini")
    else:
        print("try running the script with one of the following arguments: online, mini, validOnline, testOnline, validMini, testMini")