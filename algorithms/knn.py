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
        for i in range(length):
            distance += pow((instance1[i] - instance2[i]), 2)
        return math.sqrt(distance)

    def getNeighbors(self, trainingSet, testInstance, k):
        distances = []
        length = len(testInstance)-1
        for _, data in enumerate(trainingSet):
            dist = self.euclideanDistance(testInstance, data, length)
            distances.append((data, dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for i in range(k):
            neighbors.append(distances[i][0])
        return neighbors

    def getResponse(self, neighbors):
        classVotes = {}
        for _, neighbor in enumerate(neighbors):
            response = neighbor[-1]
            classVotes[response] = classVotes[response] + 1 if response in classVotes else 1
        sortedVotes = sorted(classVotes.items(), key = operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

    def getAccuracy(self):
        correct = 0
        for i in range(len(self.testSet)):
            if self.testSet[i][-1] == self.predictions[i][0]:
                correct += 1
        return (correct / float(len(self.testSet))) * 100.0

    def predict(self, k):
        self.predictions = []
        for _, data in enumerate(self.testSet):
            neighbors = knn.getNeighbors(self.trainingSet, data, k)
            result = knn.getResponse(neighbors)
            self.predictions.append([result, data[-1]])
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
