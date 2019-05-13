import numpy as np
class ReluLayer(object):
    def __init__(self):
        """
        Rectified Linear Unit
        """
        self.y = None

    def forward(self, x):
        """
        Implement forward pass of Relu

        y = x if x > 0
        y = 0 otherwise

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features

        Returns
        -------
        np.array
            The output of the layer

        Stores
        -------
        self.y : np.array
             The output data (need to store for backwards pass)
        """

        zero_vector = np.zeros_like(x)
        self.y = np.maximum(x, zero_vector)

        return self.y

    def backward(self, y_grad):
        """
        Implement backward pass of Relu

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """

        div_y = np.where(self.y > 0, 1, 0)

        return y_grad * div_y

    def update_param(self, lr):
        pass  # no parameters to update
