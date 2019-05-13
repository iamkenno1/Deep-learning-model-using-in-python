"""
Run least squares with provided data
"""

import numpy as np
import matplotlib.pyplot as plt
from ls import LeastSquares
import pickle


def mse(pred_train, y_train):
    diff = (pred_train - y_train) ** 2
    mse = np.sum(diff) / len(diff)
    return mse


# load data
data = pickle.load(open("ls_data.pkl", "rb"))
x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']

# run the learning algorithm
result_train = []
result_test = []
for i in range(1, 21):
    ls = LeastSquares(i)
    ls.fit(x_train, y_train)
    pred_train = ls.predict(x_train)
    pred_test = ls.predict(x_test)
    mse1 = mse(pred_train, y_train)
    mse2 = mse(pred_test, y_test)
    result_train.append(mse1)
    result_test.append(mse2)
# plot the result
plt.plot(range(1, 21), result_train, 'r', label='train_mse')
plt.plot(range(1, 21), result_test, 'b', label='test_mse')
plt.legend()
plt.grid()
plt.xlabel("Degree of the polynomial")
plt.ylabel("Mean Square Error")
plt.xticks(range(1, 21))
plt.title("Training Error and Testing Error vs Degree of the polynomial")
plt.savefig("ls_error.png")
plt.show()

# try ls
"""
ls.fit(x_train, y_train)

pred_train = ls.predict(x_train)
pred_test = ls.predict(x_test)


plt.plot(x_test, pred_test, 'r*', label='Predicted')
plt.plot(x_test, y_test, 'y*', label='Ground truth')
plt.legend()
plt.show()

"""


