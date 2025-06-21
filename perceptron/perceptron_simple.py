import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from activation_function import ActivationFunction

class PerceptronSimple:
    def __init__(self, activation_func, learning_rate=0.1, ):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.activation = activation_func

    def fit(self, X, y, max_epochs=100):
        """
        Entraîne le perceptron
        X: matrice des entrées (n_samples, n_features)
        y: vecteur des sorties désirées (n_samples,)
        """
        # Initialisation les poids et le biais
        self.weights = np.random.randn(X.shape[1])
        self.bias = 0.0

        for _ in range(max_epochs):
            for entry, result_wanted in zip(X, y):
                weighted_sum = np.dot(entry, self.weights) + self.bias
                prediction = self.activation.apply(weighted_sum)

                # Mise à jour des poids et du biais
                if prediction != result_wanted:
                    self.weights += self.learning_rate * result_wanted * entry
                    self.bias += self.learning_rate * result_wanted

    def predict(self, X):
        """Prédit les sorties pour les entrées X"""
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation.apply(linear_output)

    def score(self, X, y):
        """Calcule l'accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
if __name__ == '__main__':
    activation_functions = [
        ActivationFunction('heaviside'),
        ActivationFunction('sigmoid'),
        ActivationFunction('tanh'),
        ActivationFunction('relu'),
        ActivationFunction('leaky_relu')
    ]

    # Données pour la fonction AND
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([-1, -1, -1, 1])  # -1 pour False, 1 pour True

    for func in activation_functions:
        perceptron = PerceptronSimple(activation_func=func)
        perceptron.fit(X_and, y_and)
        print(perceptron.predict(X_and))

    print('')

    # Données pour la fonction OR
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([-1, 1, 1, 1])

    for func in activation_functions:
        perceptron = PerceptronSimple(activation_func=func)
        perceptron.fit(X_or, y_or)
        print(perceptron.predict(X_or))
