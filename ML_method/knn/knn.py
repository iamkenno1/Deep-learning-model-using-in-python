import numpy as np
from scipy import stats


class KNN(object):
    def __init__(self, k=3):
        self.x_train = None
        self.y_train = None
        self.k = k

    def fit(self, x, y):
        """
        Fit the model to the data

        For K-Nearest neighbors, the model is the data, so we just
        need to store the data

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels
        """
        self.x_train = np.copy(x)
        self.y_train = np.copy(y)



    def predict(self, x):
        """
        Predict x from the k-nearest neighbors

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            A vector of size N of the predicted class for each sample in x
        """
        ans = []

        for x_dot in x:
            # initial the list
            dis =[]
            y =[]
            # calculate the euclidean distance
            diff = (x_dot - self.x_train) ** 2
            for i in diff:
                dis.append(np.sqrt(i[0] + i[1]))
            # do the sorting and return the index
            a_sort_index = np.argsort(dis)
            # choose the first k index
            y_index = a_sort_index[0:self.k]
            # return the value from those index in y
            for index in y_index:
                y.append(self.y_train[index])
            # calculate the most common value
            result, count = stats.mode(y)

            ans.append(result[0])

        return np.asarray(ans)



