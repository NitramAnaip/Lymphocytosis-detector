import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models import CNNModel
from sklearn import metrics
from dataloader import LymphocytosisDataset


batch_size = 4
epochs = 1
model = CNNModel()

train_split = 0.8  # percentage of the data we want in train (as opposed to valdation)


transform_train = transforms.Compose(
    [
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Convert Pillow Image to Tensor
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)

train_dataset = LymphocytosisDataset(
    "/data/clinical_annotation.csv",
    "/data",
    train=True,
    valid=False,
    train_split=train_split,
    transform=transform_train,
    fill_img_list=True,
    split_label=True,
    convert_age=True,
)
print(train_dataset.__len__())

train_loader = DataLoader(train_dataset, batch_size=batch_size)

val_dataset = LymphocytosisDataset(
    "/data/clinical_annotation.csv",
    "/data",
    train=True,
    valid=True,
    train_split=train_split,
    transform=transform_train,
    fill_img_list=True,
    split_label=True,
    convert_age=True,
)

print(val_dataset.__len__())

val_loader = DataLoader(val_dataset, batch_size=batch_size)


use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Using GPU")
    model.cuda()
else:
    print("Not using CPU")

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.0001)


def train():
    model.train()
    
    conf_matrix = np.zeros((2,2))
    # nbr_batches = int(train_loader.__len__() / batch_size) + 1
    for i, (annotation, bag, targets) in enumerate(train_loader):
        #print("in batch {}".format(i))
        # start = i * batch_size
        # end = min((i + 1) * batch_size, train_loader.__len__())
        # batch_indexes = [j for j in range(start, end)]
        # batch = []
        # targets = []
        # for index in batch_indexes:
        #     # print(index)
        #     annotation, bag, target = train_loader[index]
        #     # print(type(bag))
        #     batch.append(bag)
        #     targets.append(target)

        # bag = torch.Tensor(bag)
        # print(bag.shape)
        # print(bag.permute(0,4,1,3,2).shape)
        bag = bag.permute(0, 2, 1, 3, 4)
        targets = torch.stack(targets).T
        #print(targets.shape)
        if use_cuda:
            bag, targets = bag.cuda(), targets.cuda()

        optimizer.zero_grad()
        output = model(bag)

        # pred = output.data.max(1, keepdim=True, dtype = np.float32)[1]

      

        loss = torch.nn.functional.binary_cross_entropy(output, targets.float())
        #output, targets = output.cpu().detach().numpy(), np.array(targets.float().cpu())
        #accuracy += metrics.accuracy_score(targets, output)
 
        loss.backward()
        optimizer.step()
        output = output.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        conf_matrix+=metrics.confusion_matrix(targets, output>0.5)

    return conf_matrix




def validation():
    model.eval()

    conf_matrix = np.zeros((2,2))
    for i, (annotation, bag, targets) in enumerate(val_loader):

        bag = bag.permute(0, 2, 1, 3, 4)
        targets = torch.stack(targets).T
        print(targets.shape)
        if use_cuda:
            bag, targets = bag.cuda(), targets.cuda()

        optimizer.zero_grad()
        output = model(bag)

        loss = torch.nn.functional.binary_cross_entropy(output, targets.float())
        output = output.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        conf_matrix+=metrics.confusion_matrix(targets, output>0.5)
    return conf_matrix







def main():
    train_accuracy_list = []
    val_accuracy_list = []
    for epoch in range(epochs):
        print("beginning of epoch {}".format(epoch))
        train_conf_matrix = train()
        val_conf_matrix = validation()
        train_accuracy = get_metrics(train_conf_matrix)
        val_accuracy = get_metrics(val_conf_matrix)

        print("train accuracy: {}, val acc: {}".format(train_accuracy, val_accuracy))

        train_accuracy_list.append(train_accuracy)
        val_accuracy_list.append(val_accuracy)


        print("end of epoch")


    

    if True:
        fig, ax = plt.subplots(1,1,figsize=(20,10))
        #fig.suptitle("Train-Validation accuracy", fontsize=16)
    

        ax.set_title("Train-Validation accuracy")
        ax.plot(train_accuracy_list, label='train')
        ax.plot(val_accuracy_list, label='validation')
        ax.set_xlabel('num_epochs', fontsize=12)
        ax.set_ylabel('accuracy', fontsize=12)
        ax.legend(loc='best')


      

        

        plt.savefig('./results/plots/accuracy.png')
        print('Plot saved to ' + './results/plots/accuracy.png')
    return 0

def get_metrics(conf_matrix):
    trues = conf_matrix[0, 0] + conf_matrix[1,1]
    false = conf_matrix[0,1] + conf_matrix[1,0]
    accuracy = trues/(trues + false)
    return accuracy

main()