from svm import SVM
import numpy as np
import datasets
import matplotlib.pyplot as plt

# load data
x_train, y_train, x_test, y_test = datasets.gaussian_dataset(n_train=800, n_test=800)
learning_rate = [0.01, 0.1, 1]

# run the learning algorithm
result = []
loss = []
for i in learning_rate:
    model = SVM(n_epochs=100, lr=i)

    loss.append(model.fit(x_train, y_train))

    y_pred = model.predict(x_test)

    acc = np.mean(y_pred == y_test)
    result.append(acc)
    print("logistic regression accuracy: " + str(acc))

# plot the result
plt.plot(range(0, 100), loss[0], label="learn_rate = 0.01")
plt.plot(range(0, 100), loss[1], label="learn_rate = 0.1")
plt.plot(range(0, 100), loss[2], label="learn_rate = 1")

plt.legend()
plt.grid()
plt.xlabel("iteration")
plt.ylabel("Support Vector Machine loss function value")

plt.title("loss function value vs iteration for different learning rates")
plt.savefig("svm_lr.png")
plt.show()
