import arff
import pandas as pd
import numpy as np


def splitData(data):
    rowNum, colNum = data.shape
    # 1. split data into test and train
    testSize = int(rowNum * 0.45)
    testData = data[0:testSize, :]
    trainData = data[testSize:, :]

    # 2. split test and train data into features and labels
    testFeatures = testData[:, 0:(colNum - 1)]
    testLabels = testData[:, (colNum - 1)]
    trainFeatures = trainData[:, 0:(colNum - 1)]
    trainLabels = trainData[:, (colNum - 1)]

    return testFeatures, testLabels, trainFeatures, trainLabels


def accuracy(testLabels, prediction):
    total = len(testLabels)
    goodPredict = 0
    for i in range(total):
        if testLabels[i] == prediction[i]:
            goodPredict += 1
    return (goodPredict / total) * 100


def naiveBayesClassifier(features, labels, testRow):
    classDict = {}
    trainNum = len(labels)

    # fill class dictionary with distinct classes with a initial probability of 0
    for i in labels:
        if i not in classDict:
            classDict[i] = 0

    # calculate classes probabilities and add them to class dictionary
    for _class in classDict:
        numMatching = 0
        for i in labels:
            if i == _class:
                numMatching += 1
        classDict[_class] += numMatching / trainNum

    # calculate features probabilities and add them to class dictionary
    for _class in classDict:
        featuresCol = 0
        for feature in testRow:
            matchingCount = 0
            matchingFeatures = features[labels == _class]
            matchingSize = len(matchingFeatures)
            for feat in matchingFeatures[:, featuresCol]:
                if feat == feature:
                    matchingCount += 1
            featuresCol += 1
            classDict[_class] += matchingCount / matchingSize

    # get class with maximum probability and return it
    maxProbClass = max(classDict, key=lambda x: classDict[x])
    return maxProbClass


def main():
    print('*-Naive Bayes Classifier-*')
    print('Please select a Dataset to analyze')
    while True:
        opt = input('Type 1 for Weather data, 2 for Soybean data')
        if opt not in ('1','2'):
            print('Please enter a valid option')
        else:
            break

    if int(opt) == 1:
        dataset = 'weather.nominal.arff'
    else:
        dataset = 'soybean.arff'

    # load data
    dataArff = arff.load(open(dataset, 'r', encoding='utf-8'))
    data = pd.DataFrame(dataArff['data']).dropna()
    data = np.array(data)

    # split data
    testFeatures, testLabels, trainFeatures, trainLabels = splitData(data)

    # run classifier and print results
    results = []
    for testRow in testFeatures:
        test = naiveBayesClassifier(trainFeatures, trainLabels, testRow)
        results.append(test)
    accuracyTest = accuracy(testLabels, results)
    print(dataset, ' Naive Bayes Prediction: ')
    print('Test Values: ', testLabels)
    print('Prediction: ', results)
    print(' ACCURACY: ', accuracyTest, '%')


if __name__ == '__main__':
    main()