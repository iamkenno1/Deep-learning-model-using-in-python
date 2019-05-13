import numpy as np


class SoftMaxLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.y = None

    def forward(self, x):
        """
        Implement forward pass of softmax

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
             The output of the layer (needed for backpropagation)
        """

        x_new = x - np.max(x)

        exp_x_new = np.exp(x_new)

        self.y = exp_x_new / np.transpose(np.array((exp_x_new.sum(axis=1),)))

        return self.y

    def backward(self, y_grad):
        """
        Compute "backward" computation of softmax

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        """

        ans = []
        for row in range(len(self.y)):
            j_matrix = (np.diag(self.y[row]) - np.dot(np.transpose(np.array((self.y[row],))), np.array((self.y[row],))))
            ans.append(np.dot(y_grad[row], j_matrix))
        ans_a = np.asarray(ans)

        return ans_a

    def update_param(self, lr):
        pass  # no learning for softmax layer


