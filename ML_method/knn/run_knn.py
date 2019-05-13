from knn import KNN
import numpy as np
import datasets
import matplotlib.pyplot as plt

# load data
x_train, y_train, x_test, y_test = datasets.gaussian_dataset(n_train=800, n_test=800)

# run the learning algorithm
result = []
for i in range(1, 52):

    model = KNN(k=i)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = np.mean(y_pred == y_test)
    result.append(acc)
    print("knn accuracy: " + str(acc))

# plot the result
plt.plot(result, label="accuracy")
plt.legend()
plt.grid()
plt.xlabel("parameter K")
plt.ylabel("knn accuracy")

plt.title("test accuracy on the Gaussian vs the k parameter")
plt.savefig("knn_k.png")
plt.show()
