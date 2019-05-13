import torchvision.models
import torchvision.transforms as transforms
import torch
import sklearn.svm
import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle
from util import plot_confusion_matrix
import torch.nn as nn


# get the pre-trained model
model = torchvision.models.inception_v3(pretrained=True)


# compose the pre-processing
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
resize = transforms.Resize((299, 299))
preprocessor = transforms.Compose([
    resize,
    transforms.ToTensor(),
    normalize,
])

# get the training and testing Data_loader
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder("./faces/train", preprocessor),
    batch_size=64,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder("./faces/test", preprocessor),
    batch_size=64,
    shuffle=False
)
print("Start to feature extract")
# get the training data and testing data (also the training and testing label)
all_train_label = []
all_train_data = torch.zeros(0)

# cancel the last fully connect layer
model.fc = nn.Dropout(p=0.0)

# start feature extracting
for i, (in_data, label) in enumerate(train_loader):
    with torch.no_grad():
        output, aux = model(in_data)
    all_train_data = torch.cat((all_train_data, output), 0)
    all_train_label += label.numpy().tolist()

all_test_label = []
all_test_data = torch.zeros(0)

for i, (in_data, label) in enumerate(test_loader):
    with torch.no_grad():
        output, aux = model(in_data)
    all_test_data = torch.cat((all_test_data, output), 0)
    all_test_label += label.numpy().tolist()


all_train_data = all_train_data.numpy().tolist()
all_test_data = all_test_data.numpy().tolist()


# using svm model
svm_model = sklearn.svm.SVC(C=1000, kernel='sigmoid', gamma='auto', class_weight="balanced")
svm_model.fit(all_train_data, all_train_label)
y_pred = svm_model.predict(all_test_data)
acc = np.mean(y_pred == all_test_label)
ans = ['accuracy', acc]

# print the result
with open("feature_result/feature_result.csv", "w") as result:
    writer = csv.writer(result)
    writer.writerow(y_pred)
    writer.writerow(all_test_label)
with open("feature_result/feature_extractor_accuracy.csv", "w") as result:
    writer = csv.writer(result)
    writer.writerow(ans)

plt.figure(1)
svm_title = "Feature_Extractor_SVM " + "accuracy: " + str(acc)
plot_confusion_matrix(y_pred, all_test_label, svm_title, size=9)
plt.savefig("feature_result/feature_extractor_confusion_matrix.png")
plt.show()
print("result", acc)
# save the model
# with open("feature_svm_model.pkl", "w")as file:
#     pickle.dump(svm_model, file)
