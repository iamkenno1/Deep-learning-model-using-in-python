"""
=> Your Name:Kuei Fang

In this script, you need to plot the average training loss vs epoch using a learning rate of 0.1 and a batch size of 128 for 15 epochs.

=> Final accuracy on the test set: 0.73

"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from layers.dataset import cifar100
from layers.full import FullLayer
from layers.softmax import SoftMaxLayer
from layers.cross_entropy import CrossEntropyLayer
from layers.sequential import Sequential
from layers.relu import ReluLayer
from layers.conv import ConvLayer
from layers.flatten import FlattenLayer
from layers.maxpool import MaxPoolLayer

# getting training data and testing data
(x_train, y_train), (x_test, y_test) = cifar100(seed=1213351124)

# initialize the each layer for ML model
layer1 = ConvLayer(3, 16, 3)
relu1 = ReluLayer()
maxpool1= MaxPoolLayer()
layer2 = ConvLayer(16, 32, 3)
relu2 = ReluLayer()
maxpool2 = MaxPoolLayer()
loss1 = CrossEntropyLayer()
flatten = FlattenLayer()
layer3 = FullLayer(2048, 3)
softmax1 = SoftMaxLayer()
model = Sequential(
    (
        layer1,
        relu1,
        maxpool1,
        layer2,
        relu2,
        maxpool2,
        flatten,
        layer3,
        softmax1
    ),
    loss1)

loss_epoch = []
accuracy_epoch =[]

# training and predicting
learning_rate = 0.1
loss_epoch = (model.fit(x_train, y_train, epochs=15, lr=learning_rate))
y_pred = model.predict(x_test)

acc = np.mean(y_pred == y_test)
print(str(learning_rate) + "done!!!")
print(str(acc))
accuracy_epoch.append([learning_rate, acc])

# plot the training loss vs epoch
plt.plot(range(0, 15), loss_epoch, label="learning rate = 0.1")

plt.legend()
plt.grid()
plt.xticks(np.arange(0, 15, 1))
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("training loss vs epoch for learning rates 0.1. ")
plt.savefig("loss_plot.png")

# print the accuracy in a csv file
with open("acc_result.csv", "w") as acc_result:
    writer = csv.writer(acc_result)
    for data in accuracy_epoch:
        writer.writerow(data)
"""
# print the result from prediction 
with open("pred.csv", "w") as pred_result:
    writer = csv.writer(pred_result)
    writer.writerow(y_pred)
    writer.writerow(y_test)
"""
