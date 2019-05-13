import numpy as np
import pickle

class LeastSquares(object):
    def __init__(self, k):
        """
        Initialize the LeastSquares class

        The input parameter k specifies the degree of the polynomial
        """
        self.k = k
        self.coeff = None

    def fit(self, x, y):
        """
        Find coefficients of polynomial that predicts y given x with
        degree self.k

        Store the coefficients in self.coeff
        """
        i, j = np.meshgrid(np.arange(self.k + 1), x)
        A = np.power(j, i)                              # compute the A
        A_i = np.linalg.pinv(A)                         # compute the pseudo-inverse of A
        C = np.dot(A_i, y)                              # compute the output from c = A_inverse dot y
        self.coeff = np.copy(C)

    def predict(self, x):
        """
        Predict the output given x using the learned coeffecients in
        self.coeff
        """
        i, j = np.meshgrid(np.arange(self.k +1), x)    # compute the A
        A = np.power(j, i)
        Y = np.dot(A, self.coeff)                      # compute the output from y = Ac
        return Y



