import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from activation_function import ActivationFunction

class PerceptronSimple:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y, max_epochs=100):
        """
        Entraîne le perceptron
        X: matrice des entrées (n_samples, n_features)
        y: vecteur des sorties désirées (n_samples,)
        """
        # Initialisation les poids et le biais
        self.weights = np.random.randn(X.shape[1])
        self.bias = 0.0

        activation = ActivationFunction('heaviside')

        for entry, result_wanted in zip(X, y):
            weighted_sum = np.dot(entry, self.weights)
            prediction = activation.apply(weighted_sum + self.bias)

            # Mise à jour des poids et du biais
            if prediction != result_wanted:
                self.weights += self.learning_rate * result_wanted * np.sum(entry)
                self.bias += self.learning_rate * result_wanted

    def predict(self, X):
        """Prédit les sorties pour les entrées X"""
        y_pred = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            entry = X[i] # [0, 0] [0, 1] [1, 0] [1, 1]

            # TODO

            
            y_pred[i] = 0

        return y_pred

    def score(self, X, y):
        """Calcule l'accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
if __name__ == '__main__':
    # Données pour la fonction AND
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([-1, -1, -1, 1])  # -1 pour False, 1 pour True

    perceptron = PerceptronSimple()
    perceptron.fit(X_and, y_and)
    print(perceptron.predict(X_and))


    # Données pour la fonction OR
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([-1, 1, 1, 1])

    #perceptron = PerceptronSimple()
    #perceptron.fit(X_or, y_or)
    # print(perceptron.predict(X_or))
