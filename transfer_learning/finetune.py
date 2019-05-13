
import torch
import torchvision
import torchvision.datasets
import torchvision.models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import csv

from util import plot_confusion_matrix

torch.manual_seed(0)


def get_data_loader(batchsize):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = transforms.Resize((299, 299))
    preprocessor = transforms.Compose([
        resize,
        transforms.ToTensor(),
        normalize,
    ])

    augmentation_preprocessor = transforms.Compose([
        resize,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # get the training and testing Data_loader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([torchvision.datasets.ImageFolder("./faces/train", preprocessor),
                                        torchvision.datasets.ImageFolder("./faces/train", augmentation_preprocessor)]),
        batch_size=batchsize,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder("./faces/test", preprocessor),
        batch_size=batchsize,
        shuffle=False
    )
    return train_loader, test_loader


def train_model(model, dataloaders, num_epochs, learning_r):
    model.train()
    # define loss function
    criterion = nn.CrossEntropyLoss()
    # setup SGD
    optimzer = optim.SGD(model.parameters(), lr=learning_r, momentum=0.0)

    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch, num_epochs-1))
        print('-'*10)

        running_loss =0.0
        for i, data in enumerate(dataloaders):
            print("batch: num ", i)
            # get input
            input, label = data

            # zero the parameter gradients
            optimzer.zero_grad()

            # forward
            output, aux_ouput = model(input)
            loss1 = criterion(output, label)
            loss2 = criterion(aux_ouput, label)
            loss = loss1 + 0.4*loss2

            # backward
            loss.backward()
            optimzer.step()

            running_loss += loss.item()

        print('Loss:{:.4f}'.format(running_loss / (i+1)))
    print("Finish Training")


def predict_model(model, testloader):
    # switch to evaluate mode
    model.eval()
    correct = 0
    total = 0
    all_predicted = []
    # without autograd can save some memory
    with torch.no_grad():
        for images, labels in testloader:
            # forward
            outputs = model(images)

            # compute the performance
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predicted += predicted.numpy().tolist()
        # print the result
        print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
        acc = (100*correct / total)
    return all_predicted, acc





if __name__ == '__main__':
    # get data

    batch_size = 16
    num_classes = 31
    epoch_num = 1
    learning_rate = 0.1
    trainloader, testloader = get_data_loader(batch_size)

    print(len(trainloader))

    # fine tune
    print "Fine tuning"
    model = torchvision.models.inception_v3(pretrained=True)
    model.AuxLogits.fc = nn.Linear(768, num_classes)
    model.fc = nn.Linear(2048, num_classes)
    train_model(model, trainloader, num_epochs=epoch_num, learning_r=learning_rate)
    torch.save(model, "finetune_result/fine_tune_model.pth")
    pred_labels, acc = predict_model(model, testloader)
    test_labels = testloader.dataset.targets

    # plot the confusion matrix
    plt.figure(1)
    title = "Fine_tune " + "Accuracy: " + str(acc) + ' %'
    plot_confusion_matrix(pred_labels, test_labels, title, size=9)
    plt.savefig("finetune_result/conf_finetune.png")
    plt.show()
    # print the result
    print(acc)
    r_acc =["Accuracy", acc]
    with open("finetune_result/finetune_result.csv", "w") as result:
        writer = csv.writer(result)
        writer.writerow(pred_labels)
        writer.writerow(test_labels)
    with open("finetune_result/finetune_accuracy.csv", "w") as ans:
        writer = csv.writer(ans)
        writer.writerow(r_acc)
    torch.save(model, "finetune_result/fine_tune_model_augmentation.pth")


