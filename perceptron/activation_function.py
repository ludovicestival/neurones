import numpy as np
import matplotlib.pyplot as plt

class ActivationFunction:
    def __init__(self, name, alpha=0.01):
        self.name = name.lower()
        self.alpha = alpha  # Pour Leaky ReLU

    def apply(self, z):
        if self.name == "heaviside":
            return z >= 0
        elif self.name == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif self.name == "tanh":
            return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        elif self.name == "relu":
            return np.where(z < 0, 0, z)
        elif self.name == "leaky_relu":
            return np.where(z < 0, self.alpha * z, z)
        else:
            raise ValueError(f"Activation '{self.name}' non reconnue.")

    def derivative(self, z):
        if self.name == "heaviside":
            # La dérivée de Heaviside est la distribution de Dirac
            return 0
        elif self.name == "sigmoid":
            return np.exp(z) / ((np.exp(-z)+1)**2)
        elif self.name == "tanh":
            return 1 - (np.exp(z)-np.exp(-z))**2 / (np.exp(z)+np.exp(-z))**2
        elif self.name == "relu":
            return np.where(z < 0, 0, 1)
        elif self.name == "leaky_relu":
            return np.where(z < 0, z, 1)
        else:
            raise ValueError(f"Dérivée de '{self.name}' non définie.") 

if __name__ == '__main__':
    z = np.linspace(-10, 10, 1000)

    for name in ['heaviside', 'sigmoid', 'tanh', 'relu', 'leaky_relu']:
        result1 = ActivationFunction(name).apply(z)
        result2 = ActivationFunction(name).derivative(z)
        plt.figure()
        ax = plt.gca()
        plt.plot(result1, label=name)
        plt.plot(result2, label='dérivée')
        plt.legend()
        #plt.show()
        plt.savefig('figures/' + name + '.png')
