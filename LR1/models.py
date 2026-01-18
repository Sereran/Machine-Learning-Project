import numpy as np

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, threshold=0.5):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.threshold = threshold
        self.weights = None
        self.bias = None
        self.losses = []
        
    def _sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def _loss(self, y, y_hat):
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []
        
        for i in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            loss = self._loss(y, y_predicted)
            self.losses.append(loss)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict_prob(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
    
    def predict(self, X):
        y_predicted_cls = [1 if i > self.threshold else 0 for i in self.predict_prob(X)]
        return np.array(y_predicted_cls)