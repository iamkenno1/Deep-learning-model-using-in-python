import numpy as np


class SVM(object):
    def __init__(self, n_epochs=10, lr=0.1, l2_reg=1):
        """
        """
        self.b = None
        self.w = None
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2_reg = l2_reg

    def forward(self, x):
        """
        Compute "forward" computation of SVM f(x) = w^T + b

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


        a = np.dot(self.w, np.transpose(x)) + self.b
        ans = a.flatten()


        return ans

    def loss(self, x, y):
        """
        Return the SVM hinge loss
        L(x) = max(0, 1-y(f(x))) + 0.5 w^T w

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
        a = 1-y*self.forward(x)
        length = len(a)
        zero_vector = np.zeros(length)
        a = np.maximum(a, zero_vector)


        result = a.sum()/len(y) + self.l2_reg * np.dot(self.w, np.transpose(self.w)) * 0.5

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
        a = 1 - y * self.forward(x)
        result = 0
        for i in range(len(a)):
            if a[i] > 0:
                result += (-1)*y[i]

        return float(result)/len(y)

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
        a = 1 - y * self.forward(x)
        result = 0
        for i in range(len(a)):
            if a[i] > 0:
                result += (-1) * y[i] * x[i]
        return result / len(y) + self.l2_reg * self.w

    def fit(self, x, y, plot=False):
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
            if i > 0:
                ans.append(1)
            else:
                ans.append(-1)

        return ans
