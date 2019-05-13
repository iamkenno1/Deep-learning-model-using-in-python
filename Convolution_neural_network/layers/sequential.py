from __future__ import print_function
import numpy as np
from layers.full import FullLayer
from layers.softmax import SoftMaxLayer
from layers.cross_entropy import CrossEntropyLayer
from layers.relu import ReluLayer



class Sequential(object):
    def __init__(self, layers, loss):
        """
        Sequential model

        Implements a sequence of layers

        Parameters
        ----------
        layers : list of layer objects
        loss : loss object
        """
        self.layers = layers
        self.loss = loss

    def forward(self, x, target=None):
        """
        Forward pass through all layers
        
        if target is not none, then also do loss layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features
        target : np.array
            The target data of size number of training samples x number of features (one-hot)

        Returns
        -------
        np.array
            The output of the model
        """
        forward_input = np.copy(x)
        for layer in self.layers:
            forward_input = layer.forward(forward_input)
        try:
            if len(target) > 0:
                loss1 = self.loss.forward(forward_input, target)
                return loss1
        except:
            return forward_input


    def backward(self):
        """
        Compute "backward" computation of fully connected layer

        Returns
        -------
        np.array
            The gradient at the input

        """

        output = self.loss.backward()

        for layer in self.layers[-1::-1]:
            output = layer.backward(output)

        return output


    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate
        """
        for layer in self.layers:
            layer.update_param(lr)

    def fit(self, x, y, epochs=10, lr=0.1, batch_size=128):
        """
        Fit parameters of all layers using batches

        Parameters
        ----------
        x : numpy matrix
            Training data (number of samples x number of features)
        y : numpy matrix
            Training labels (number of samples x number of features) (one-hot)
        epochs: integer
            Number of epochs to run (1 epoch = 1 pass through entire data)
        lr: float
            Learning rate
        batch_size: integer
            Number of data samples per batch of gradient descent
        """
        length = len(x)
        batch_number = length/batch_size
        redundant = length % batch_size
        loss_epoch = np.zeros(epochs)

        for i in xrange(epochs):
            print("epoch done")
            loss_batch = 0
            for j in xrange(batch_number):
                loss_batch += self.forward(x[j*128:(j+1)*128], target=y[j*128:(j+1)*128])
                self.backward()
                self.update_param(lr)
            if redundant > 0:
                loss_batch += self.forward(x[(j+1)*128:], target=y[(j+1)*128:])
                self.backward()
                self.update_param(lr)
                loss_batch /= (batch_number+1)
            else:
                loss_batch /= batch_number
            loss_epoch[i] = loss_batch
        return loss_epoch

    def predict(self, x):
        """
        Return class prediction with input x

        Parameters
        ----------
        x : numpy matrix
            Testing data data (number of samples x number of features)

        Returns
        -------
        np.array
            The output of the model (integer class predictions)
        """
        result = self.forward(x)
        ans = np.argmax(result, axis=1)

        return ans









