import numpy as np

class FlattenLayer(object):
    def __init__(self):
        """
        Flatten layer
        """
        self.orig_shape = None # to store the shape for backpropagation

    def forward(self, x):
        """
        Compute "forward" computation of flatten layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the flatten operation
            size = training samples x (number of input channels * number of rows * number of columns)
            (should make a copy of the data with np.copy)

        Stores
        -------
        self.orig_shape : list
             The original shape of the data
        """

        self.orig_shape = [len(x[0]), len(x[0][0]), len(x[0][0][0])]

        return np.copy(np.reshape(x, (len(x), len(x[0][0][0])*len(x[0][0])*len(x[0]))))



    def backward(self, y_grad):
        """
        Compute "backward" computation of flatten layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        return np.copy(np.reshape(y_grad, (len(y_grad), self.orig_shape[0], self.orig_shape[1], self.orig_shape[2])))



    def update_param(self, lr):
        pass
