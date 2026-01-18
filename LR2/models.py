import numpy as np


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def get_coeffs(self):
        return self.weights

    def _sigmoid(self, z):
        # Maps any number to a probability between 0 and 1
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent Loop
        for _ in range(self.n_iterations):
            # Linear model: z = w*x + b
            linear_model = np.dot(X, self.weights) + self.bias
            # Prediction (Probability)
            y_predicted = self._sigmoid(linear_model)

            # Compute Gradients (derivatives)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        y_predicted_cls = [1 if i > threshold else 0 for i in self.predict_proba(X)]
        return np.array(y_predicted_cls)


class NaiveBayesRanking:
    def __init__(self):
        self.probs_given_cart = {}  # P(Product | Cart Feature)
        self.priors = {}  # P(Product)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        # Calculate mean and variance for each feature per class
        self.mean = X.groupby(y).mean()
        self.var = X.groupby(y).var()
        self.priors = X.groupby(y).count().iloc[:, 0] / n_samples

    def _calculate_likelihood(self, class_idx, x):
        # Gaussian probability density function
        mean = self.mean.iloc[class_idx]
        var = self.var.iloc[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict_proba(self, X):
        # Returns probability of class 1 (Buying the item)
        y_pred = []
        for x in X.values:
            # Calculate posterior probability for class 1
            posterior_1 = np.log(self.priors[1]) + np.sum(np.log(self._calculate_likelihood(1, x) + 1e-9))
            posterior_0 = np.log(self.priors[0]) + np.sum(np.log(self._calculate_likelihood(0, x) + 1e-9))

            # Convert back from log scale
            y_pred.append(posterior_1)
        return np.array(y_pred)