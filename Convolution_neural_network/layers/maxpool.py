import numpy as np


class MaxPoolLayer(object):
    def __init__(self, size=2):
        """
        MaxPool layer
        Ok to assume non-overlapping regions
        """
        self.locs = None  # to store max locations
        self.size = size  # size of the pooling

    def forward(self, x):
        """
        Compute "forward" computation of max pooling layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the maxpooling

        Stores
        -------
        self.locs : np.array
             The locations of the maxes (needed for back propagation)
        """
        # getting constant
        batch_size = len(x)
        channel_size = len(x[0])
        row =len(x[0][0])
        col = len(x[0][0][0])

        row_stride = row//self.size
        col_stride = col//self.size
        self.locs = np.zeros_like(x)

        # pooling
        ans =[]
        for i in range(batch_size):
            result=[]
            for j in range(channel_size):
                output_array = np.zeros((row_stride, col_stride), )
                for k in range(row_stride):
                    for l in range(col_stride):
                        start_row = k * self.size
                        start_col = l * self.size
                        pool = x[i][j][start_row:start_row+self.size, start_col:start_col+self.size]
                        output_array[k][l] = pool.max()
                        zero_pool = (pool == pool.max())
                        pool = zero_pool.astype(int)
                        self.locs[i][j][start_row:start_row+self.size, start_col:start_col+self.size] = np.copy(pool)
                result.append(output_array)
            ans.append(np.asarray(result))

        return np.asarray(ans)

    def backward(self, y_grad):
        """
        Compute "backward" computation of maxpool layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        # getting constant
        batch_size = len(self.locs)
        channel_size = len(self.locs[0])
        row = len(self.locs[0][0])
        col = len(self.locs[0][0][0])

        row_stride = row // self.size
        col_stride = col // self.size
        result = np.zeros_like(self.locs)

        # backpropagation of pooling
        for i in range(batch_size):
            for j in range(channel_size):
                for k in range(row_stride):
                    for l in range(col_stride):
                        start_row = k * self.size
                        start_col = l * self.size
                        result[i][j][start_row:start_row + 2, start_col:start_col + 2] = self.locs[i][j][start_row:start_row + 2, start_col:start_col + 2] * y_grad[i][j][k][l]

        return result



    def update_param(self, lr):
        pass
