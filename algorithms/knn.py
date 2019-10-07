import csv
import random
import math
import operator


class KNClassifier:

    def __init__(self):
        self.dataSet = []
        self.trainingSet = []
        self.testSet = []
        self.predictions = []

    def loadDataset(self, filename, label):
        with open(filename, 'rt') as csvfile:
            lines = csv.reader(csvfile)
            dataset = list(lines)
            for c, val in enumerate(dataset):
                for i, data in enumerate(val):
                    if i != (label - 1):
                        dataset[c][i] = float(data)
            self.dataSet = dataset

    def splitTestAndTrainData(self, proportion=0.5):
        for _, data in enumerate(self.dataSet):
            if random.random() < proportion:
                self.trainingSet.append(data)
            else:
                self.testSet.append(data)

    def euclideanDistance(self, instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)

    def getNeighbors(self, trainingSet, testInstance, k):
        distances = []
        length = len(testInstance)-1
        for x in range(len(trainingSet)):
            dist = self.euclideanDistance(testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def getResponse(self, neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(),
                             key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

    def getAccuracy(self):
        correct = 0
        for x in range(len(self.testSet)):
            if self.testSet[x][-1] == self.predictions[x][0]:
                correct += 1
        return (correct / float(len(self.testSet))) * 100.0

    def predict(self, k):
        self.predictions = []
        for x in range(len(self.testSet)):
            neighbors = knn.getNeighbors(self.trainingSet, self.testSet[x], k)
            result = knn.getResponse(neighbors)
            self.predictions.append([result, self.testSet[x][-1]])
        return self.predictions


if __name__ == "__main__":
    knn = KNClassifier()
    knn.loadDataset('data/iris.csv', label=5)
    knn.splitTestAndTrainData(proportion=0.66)
    predictions = knn.predict(3)

    for x in predictions:
        print('> predicted=' + repr(x[0]), '> actual=' + repr(x[1]))

    accuracy = knn.getAccuracy()
    print("Accuracy: " + repr(accuracy) + "%")
