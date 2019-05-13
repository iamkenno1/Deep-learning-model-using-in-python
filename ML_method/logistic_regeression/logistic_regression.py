import numpy as np


class LogisticRegression(object):
    def __init__(self, n_epochs=10, lr=0.1, l2_reg=0):
        """
        Initialize variables
        """
        self.b = None
        self.w = None
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2_reg = l2_reg

    def forward(self, x):
        """
        Compute "forward" computation of logistic regression

        This will return the squashing function:
        f(x) = 1 / (1 + exp(-(w^T x + b)))

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            A 1 dimensional vector of the logistic function
        """
        a = 1/(1 + np.exp(-1*(np.dot(self.w, np.transpose(x)) + self.b)))
        ans = a.flatten()

        return ans

    def loss(self, x, y):
        """
        Return the logistic loss
        L(x) = 1/N * (ln(1 + exp(-y * (w^Tx + b)))) + 1/2 * lambda * w^T * w

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        float
            The logistic loss value
        """
        matrix = np.dot(self.w, np.transpose(x)) + self.b

        result = np.log(1 + np.exp(-1*y * matrix))

        result = np.sum(result) / len(y) + self.l2_reg * np.dot(self.w, np.transpose(self.w))/2

        return result.sum()

    def grad_loss_wrt_b(self, x, y):
        """
        Compute the gradient of the loss with respect to b

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        float
            The gradient
        """
        result = (np.dot(self.w, np.transpose(x)) + self.b)
        total = (-1)*y / (1 + np.exp(y * result))
        total = np.sum(total) / len(y)

        return total

    def grad_loss_wrt_w(self, x, y):
        """
        Compute the gradient of the loss with respect to w

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        np.array
            The gradient (should be the same size as self.w)
        """

        result = (np.dot(self.w, np.transpose(x)) + self.b)
        total = (-1) * y * np.transpose(x) / (1 + np.exp(y * result))

        ans = np.sum(total, axis=1)/len(y) + self.l2_reg * self.w

        return ans

    def fit(self, x, y):
        """
        Fit the model to the data

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels
        """
        length = len(np.transpose(x))
        self.w = np.random.rand(length)
        self.b = 0
        loss_value = []
        for i in range(self.n_epochs):
            self.b -= self.lr * self.grad_loss_wrt_b(x, y)
            self.w -= self.lr * self.grad_loss_wrt_w(x, y)
            loss_value.append(self.loss(x, y))
        return loss_value

    def predict(self, x):
        """
        Predict the labels for test data x

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            Vector of predicted class labels for every training sample
        """
        ans = []
        result = self.forward(x)

        for i in result:
            if i > 0.5:
                ans.append(1)
            else:
                ans.append(-1)

        return ans





