import numpy as np
import scipy.signal


class ConvLayer(object):
    def __init__(self, n_i, n_o, h):
        """
        Convolutional layer

        Parameters
        ----------
        n_i : integer
            The number of input channels
        n_o : integer
            The number of output channels
        h : integer
            The size of the filter
        """
        # glorot initialization
        self.b = np.array(np.zeros(n_o,), 'float64')
        self.W = np.random.randn(n_o, n_i, h, h) * np.sqrt(np.sqrt(2/float((n_i+n_o)*h*h)))
        self.n_i = n_i
        self.n_o = n_o

        self.W_grad = None
        self.b_grad = None

    def forward(self, x):
        """
        Compute "forward" computation of convolutional layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the convolutiona

        Stores
        -------
        self.x : np.array
             The input data (need to store for backwards pass)

        """
        # getting data size

        batch_size = len(x)
        filter_size = len(self.W[0][0][0])
        padding_size = filter_size//2
        row = len(x[0][0])
        col = len(x[0][0][0])

        # initialize
        forward_result = [[0]*self.n_o]*batch_size
        ans =[]

        # store
        self.x = np.copy(x)

        # loop for each batch

        for i in range(batch_size):
            for k in range(self.n_o):
                correlation_sum = np.zeros((row, col),)
                for j in range(self.n_i):
                    # padding
                    test = np.pad(x[i][j], ((padding_size, padding_size), (padding_size, padding_size)),
                                  'constant', constant_values=0)
                    # correlate
                    correlation_sum += scipy.signal.correlate(test, self.W[k][j], mode='valid')

                forward_result[i][k] = np.copy(correlation_sum) + self.b[k]
            ans.append(np.asarray(forward_result[i]))
        return np.asarray(ans)

    def backward(self, y_grad):
        """
        Compute "backward" computation of convolutional layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        Stores
        -------
        self.b_grad : np.array
             The gradient with respect to b (same dimensions as self.b)
        self.w_grad : np.array
             The gradient with respect to W (same dimensions as self.W
        """
        # getting data size
        batch_size = len(y_grad)
        padding_size = len(self.W[0][0][0])//2
        # update b_grad
        b_grad = np.zeros((self.n_o,), )
        for i in range(self.n_o):
            sum_value =0
            for j in range(batch_size):
                sum_value += y_grad[j][i].sum()
            b_grad[i] = sum_value
        self.b_grad = np.copy(b_grad)
        # update w_grad
        ans =np.zeros_like(self.W)
        for i in range(batch_size):
            W_grad = []
            for k in range(self.n_o):
                channel = []
                for j in range(self.n_i):
                    padding = np.pad(self.x[i][j], ((padding_size, padding_size), (padding_size, padding_size)), 'constant', constant_values=0)
                    channel.append(scipy.signal.correlate(padding, y_grad[i][k], mode='valid'))
                W_grad.append(channel)
            ans += (np.asarray(W_grad))
        self.W_grad = np.asarray(ans)

        # update x_grad
        x_grad =[]
        for i in range(batch_size):
            out_channel=[]
            for j in range(self.n_i):
                sum_channel = np.zeros_like(self.x[0][0])
                for k in range(self.n_o):
                    padding = np.pad(y_grad[i][k], ((padding_size, padding_size), (padding_size, padding_size)),
                                     'constant', constant_values=0)
                    sum_channel += scipy.signal.convolve(padding, self.W[k][j], mode='valid')
                out_channel.append(sum_channel)
            x_grad.append(out_channel)

        return np.asarray(x_grad)

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate

        Stores
        -------
        self.W : np.array
             The updated value for self.W
        self.b : np.array
             The updated value for self.b
        """
        self.W -= self.W_grad*lr
        self.b -= self.b_grad*lr

