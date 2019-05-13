import numpy as np


class CrossEntropyLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.x = None
        self.t = None

    def forward(self, x, t):
        """
        Implements forward pass of cross entropy

        l(x,t) = -1/N * sum(log(x) * t)

        where
        x = input (number of samples x feature dimension)
        t = target with one hot encoding (number of samples x feature dimension)
        N = number of samples (constant)

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x feature dimension
        t : np.array
            The target data (one-hot) of size number of training samples x feature dimension

        Returns
        -------
        np.array
            The output of the loss

        Stores
        -------
        self.x : np.array
             The input data (need to store for backwards pass)
        self.t : np.array
             The target data (need to store for backwards pass)
        """
        self.x = np.copy(x)
        self.t = np.copy(t)
        return np.array(((-1 * np.sum(np.log(x)*t)/len(x)), ))


    def backward(self, y_grad=None):
        """
        Compute "backward" computation of softmax loss layer

        Returns
        -------
        np.array
            The gradient at the input

        """

        return -1*self.t/self.x/len(self.x)

