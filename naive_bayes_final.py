import arff
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math


def naiveBayesClassifier(features, labels, check):
    classNum = []
    for i in labels:
        if i not in classNum:
            classNum.append(i)
    numLabels = len(classNum)
    trainRows, trainCols = features.shape
    probabilities = [0] * numLabels
    NumClassEnum = enumerate(classNum)
    for (i, label) in NumClassEnum:
        clasProb = len(labels[labels == label]) / trainRows
        checkEnum = enumerate(check)
        for (j, feature) in checkEnum:
            trainFeatures = features[labels == label]
            count = 1e-1
            for feat in trainFeatures[:, j]:
                if feat == feature:
                    count += 1
            probabilities[i] += math.log(count / len(trainFeatures))
        probabilities[i] += math.log(clasProb)
    maxVal = np.argmax(probabilities)
    return classNum[int(maxVal)]


def main():
    # Initiation and selection of dataset to analyze
    print('*** Welcome to Naive Bayes Classifier ***')
    print('Please select a Dataset to analyze:')
    while True:
        opt = input('Type 1 for Weather data, 2 for Soybean data: ')
        if opt not in ('1', '2'):
            print('Please enter a valid option: ')
        else:
            break

    if int(opt) == 1:
        dataset = 'weather.nominal.arff'
    else:
        dataset = 'soybean.arff'

    # load data
    dataArff = arff.load(open(dataset, 'r', encoding="utf-8"))
    attrs = dataArff['attributes']
    data = pd.DataFrame(dataArff['data']).dropna()
    data = np.array(data)

    atrrLen = len(attrs)
    classPos = attrs[atrrLen - 1][0]
    dataLabels = data[:, (atrrLen-1)]
    dataFeatures = data[:, 0:(atrrLen-1)]

    # split data into train and test
    features_train, features_test, labels_train, labels_test = train_test_split(dataFeatures, dataLabels, test_size=0.4,
                                                                                random_state=100)
    # run classifier & print results
    result = []
    for testRow in features_test:
        result.append(naiveBayesClassifier(features_train, labels_train, testRow))
    accuracyTest = accuracy_score(labels_test, result) * 100
    print('\n', dataset, ' Dataset Naive Bayes prediction:')
    print('Test Values: ', labels_test)
    print('Prediction: ', np.array(result))
    print('\n***Test Accuracy: ', accuracyTest, '***')


if __name__ == '__main__':
    main()