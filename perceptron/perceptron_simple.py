import numpy as np
import matplotlib.pyplot as plt

#from tqdm import tqdm

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
        # TODO: Initialiser les poids et le biais
        # TODO: Implémenter l'algorithme d'apprentissage
        pass
    
    def predict(self, X):
        """Prédit les sorties pour les entrées X"""
        # TODO: Calculer les prédictions
        pass
    
    def score(self, X, y):
        """Calcule l'accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)