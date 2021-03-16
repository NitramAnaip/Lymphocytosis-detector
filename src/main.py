import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models import CNNModel, TransferUNet, Attention
from sklearn import metrics
from dataloader import LymphocytosisDataset







# Training settings
parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--batch-size', type=int, default=4, metavar='B',
                    help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.01, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--train-split', type=str, default=0.8, metavar='E',
                    help='percentage of data to use as train.')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)



batch_size = args.batch_size
epochs = args.epochs
model = Attention()
train_split = args.train_split  # percentage of the data we want in train (as opposed to valdation)




transform_train = transforms.Compose(
    [
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),  # Convert Pillow Image to Tensor
        # transforms.Resize(128)
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
    convert_gender=True,
)
print(train_dataset.__len__())

train_loader = DataLoader(train_dataset, batch_size=batch_size)

val_dataset = LymphocytosisDataset(
    "/data/clinical_annotation.csv",
    "/data",
    train=True,
    valid=True,
    train_split=train_split,
    #transform=transform_train, no need for transformation on validation set
    fill_img_list=True,
    split_label=True,
    convert_age=True,
    convert_gender=True,
)

print(val_dataset.__len__())

val_loader = DataLoader(val_dataset, batch_size=batch_size)


use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Using GPU")
    model.cuda()
else:
    print("Not using CPU")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.0001)


def train(permute=True):
    model.train()

    conf_matrix = np.zeros((2, 2))
    for i, (annotation, bag, targets) in enumerate(tqdm(train_loader)):
        # print("in batch {}".format(i))
        if permute:
            bag = bag.permute(0, 2, 1, 3, 4)
        targets = torch.stack(targets).T
        if use_cuda:
            bag, targets = bag.cuda(), targets.cuda()
            for key, value in annotation.items():
                annotation[key] = value.cuda()

        optimizer.zero_grad()
        output = model(bag, annotation)


        loss = torch.nn.functional.binary_cross_entropy(output, targets.float())

        loss.backward()
        optimizer.step()
        output = output.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        conf_matrix += metrics.confusion_matrix(targets, output > 0.5)

    return conf_matrix


def validation(permute=True):
    model.eval()

    conf_matrix = np.zeros((2, 2))
    for i, (annotation, bag, targets) in enumerate(val_loader):

        if permute:
            bag = bag.permute(0, 2, 1, 3, 4)
        targets = torch.stack(targets).T
        if use_cuda:
            bag, targets = bag.cuda(), targets.cuda()
            for key, value in annotation.items():
                annotation[key] = value.cuda()

        optimizer.zero_grad()
        output = model(bag, annotation)

        loss = torch.nn.functional.binary_cross_entropy(output, targets.float())
        output = output.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        conf_matrix += metrics.confusion_matrix(targets, output > 0.5)
    return conf_matrix


def main():
    metrics = {"train": [[], [], [], []], "valid": [[], [], [], []]}
    metric_names = ["accuracy", "recall", "precision", "f1"]

    for epoch in range(epochs):
        print("beginning of epoch {}".format(epoch))
        train_conf_matrix = train(permute=True)
        val_conf_matrix = validation(permute=True)
        train_metrics = get_metrics(train_conf_matrix)
        val_metrics = get_metrics(val_conf_matrix)

        print(
            "train accuracy: {}, val acc: {}".format(train_metrics[0], val_metrics[0])
        )

        for i in range(len(metrics["train"])):
            metrics["train"][i].append(train_metrics[i])
            metrics["valid"][i].append(val_metrics[i])

        print("end of epoch")

    if True:
        nbr_subplots = 4
        fig, ax = plt.subplots(1, nbr_subplots, figsize=(20, 10))
        # fig.suptitle("Train-Validation accuracy", fontsize=16)

        for i in range(nbr_subplots):
            ax[i].set_title("Train-Validation {}".format(metric_names[i]))
            ax[i].plot(metrics["train"][i], label="train")
            ax[i].plot(metrics["valid"][i], label="validation")
            ax[i].set_xlabel("num_epochs", fontsize=12)
            ax[i].set_ylabel(metric_names[i], fontsize=12)
            ax[i].legend(loc="best")

        plt.savefig("./results/plots/Attention.png")
        print("Plot saved to " + "./results/plots/Attention.png")
    return 0


def get_metrics(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = (2 * recall * precision) / (recall + precision)
    return [accuracy, recall, precision, f1]


main()