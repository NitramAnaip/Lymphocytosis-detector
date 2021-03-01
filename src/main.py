import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import CNNModel
from dataloader import LymphocytosisDataset


batch_size = 2
epochs = 10
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
    "../data/clinical_annotation.csv",
    "../data",
    train=True,
    valid=False,
    train_split=train_split,
    transform=transform_train,
    fill_img_list=True,
    split_label=True,
    convert_age=True,
)

train_loader = DataLoader(train_dataset, batch_size=batch_size)

val_dataset = LymphocytosisDataset(
    "../data/clinical_annotation.csv",
    "../data",
    train=True,
    valid=True,
    train_split=train_split,
    transform=transform_train,
    fill_img_list=True,
    split_label=True,
    convert_age=True,
)

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
    # nbr_batches = int(train_loader.__len__() / batch_size) + 1
    for i, (annotation, bag, targets) in enumerate(train_loader):
        print("in batch {}".format(i))
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
        targets = torch.stack(targets)
        print(targets.shape)
        if use_cuda:
            bag, targets = bag.cuda(), targets.cuda()

        optimizer.zero_grad()
        output = model(bag)

        # pred = output.data.max(1, keepdim=True, dtype = np.float32)[1]

        print(output.size())
        print(targets)
        print(targets.size())

        loss = torch.nn.functional.binary_cross_entropy(output, targets)
        loss.backward()
        optimizer.step()

    return 0


def validation():
    model.eval()
    nbr_batches = int(val_loader.__len__() / batch_size) + 1
    for i in range(nbr_batches):
        print("in batch {}".format(i))
        start = i * batch_size
        end = min((i + 1) * batch_size, val_loader.__len__())
        batch_indexes = [j for j in range(start, end)]
        batch = []
        targets = []
        for index in batch_indexes:
            annotation, bag, target = val_loader[index]
            batch.append(bag)
            targets.append(target)

        batch = torch.Tensor(batch)
        batch = batch.permute(0, 4, 1, 3, 2)
        targets = torch.Tensor(targets)
        if use_cuda:
            batch, targets = batch.cuda(), targets.cuda()

        optimizer.zero_grad()
        output = model(batch)
        print(output.size())
        print(targets.size())

        loss = torch.nn.functional.binary_cross_entropy(output, targets)

    return correct / len(val_loader.dataset), corrects


def main():
    for epoch in range(epochs):
        print("beginning of epoch {}".format(epoch))
        train()

        print("end of epoch")
    return 0


main()