from algorithms.knn import KNClassifier

# Testing accuracy of classification
def adult_data_knn_accuracy():
    knn = KNClassifier()
    knn.load_dataset('data/adult.csv', label=15, batch=1, titles=True)

    # Test prediction accuracy
    knn.split_data(proportion=0.66)
    predictions = knn.predict(k=3)
    for x in predictions:
        print('> predicted=' + repr(x[0]), '> actual=' + repr(x[1]))
    accuracy = knn.get_accuracy()
    print("Accuracy: " + repr(accuracy) + "%")

# Predicting data with unknown values
def adult_data_knn_predict():
    knn = KNClassifier()
    knn.load_dataset('data/adult.csv', label=15, batch=1, titles=True)
    knn.load_data_for_predict('data/unpredicted_adults.csv', titles=True)
    predictions = knn.predict(k=3)
    for x in predictions:
        print('> predicted=' + repr(x[0]), '> actual=' + repr(x[1]))

def imdb_data_knn_predict():
    knn = KNClassifier()
    knn.load_dataset('data/imdb.csv', label=2, batch=1, titles=True)
    knn.load_data_for_predict('data/unpredicted_imdb.csv', titles=True)
    predictions = knn.predict(k=3)
    for x in predictions:
        print('> predicted=' + repr(x[0]), '> actual=' + repr(x[1]))


if __name__ == "__main__":
    imdb_data_knn_predict()
