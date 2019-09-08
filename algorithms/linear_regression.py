# imports
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self, learning_rate=0.05, iteration=1000):
        self.learning_rate = learning_rate
        self.iteration = iteration

    def fit(self, x, y):
        """
                Fit the training data
        Parameters
        ----------
        x: array
            Training samples
        y: array
            Target values

        Returns
        -------
        self : object
        """

        self.cost_ = []
        self.w_ = np.zeros((x.shape[1], 1))

        # of rows
        m = x.shape[0]

        for _ in range(self.iteration):
            y_predict = np.dot(x, self.w_)
            residuals = y_predict - y
            gradient_vector = np.dot(x.T, residuals)
            self.w_ -= (self.learning_rate/m) * gradient_vector
            cost = np.sum((residuals ** 2)) / (2*m)
            self.cost_.append(cost)

        return self

    def predict(self, x):
        return np.dot(x, self.w_)

if __name__ == "__main__":
    # generate random data-set
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 2 + 3 * x + np.random.rand(100, 1)
    reg = LinearRegression()
    reg.fit(x, y)
    y_predicted = reg.predict(x)

    # plot
    plt.scatter(x, y, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y_predicted, color='r')
    plt.show()