import numpy as np
import torch
from models import CNNModel
from dataloader import LymphocytosisDataset


batch_size = 2
epochs = 10
model = CNNModel()

train_split = 0.8 # percentage of the data we want in train (as opposed to valdation)

train_loader = LymphocytosisDataset("/data/clinical_annotation.csv",
        "/data",
        train = True,
        valid = False, 
        train_split = train_split, 
        transform = None,
        fill_img_list = True,
        split_label = True)

val_loader = LymphocytosisDataset("/data/clinical_annotation.csv",
        "/data",
        train = True,
        valid = True, 
        train_split = train_split, 
        transform = None,
        fill_img_list = True,
        split_label = True)


use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.0001)



def train():
    model.train()
    nbr_batches = int(train_loader.__len__()/batch_size)+1
    for i in range (nbr_batches):
        print("in batch {}".format(i))
        start = i*batch_size
        end = min( (i+1)*batch_size, train_loader.__len__() )
        batch_indexes = [j for j in range (start, end)]
        batch = []
        targets = []
        for index in batch_indexes:
            #print(index)
            annotation, bag, target = train_loader[index]
            #print(type(bag))
            batch.append(bag)
            targets.append(target)
        
        batch = torch.Tensor(batch)
        #print(batch.shape)
        #print(batch.permute(0,4,1,3,2).shape)
        batch = batch.permute(0,4,1,3,2)
        targets = torch.Tensor(targets)
        print(targets.shape)
        if use_cuda:
            batch, targets = batch.cuda(), targets.cuda()

        optimizer.zero_grad()
        output = model(batch)

        #pred = output.data.max(1, keepdim=True, dtype = np.float32)[1]

        print(output.size())
        print(targets)
        print(targets.size())

        loss = torch.nn.functional.binary_cross_entropy(output, targets)
        loss.backward()
        optimizer.step()

    return 0


def validation():
    model.eval()
    nbr_batches = int(val_loader.__len__()/batch_size) + 1
    for i in range (nbr_batches):
        print("in batch {}".format(i))
        start = i*batch_size
        end = min( (i+1)*batch_size, val_loader.__len__() )
        batch_indexes = [j for j in range (start, end)]
        batch = []
        targets = []
        for index in batch_indexes:
            annotation, bag, target = val_loader[index]
            batch.append(bag)
            targets.append(target)
        
        batch = torch.Tensor(batch)
        batch = batch.permute(0,4,1,3,2)
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
    for epoch in range (epochs):
        print("beginning of epoch {}".format(epoch))
        train()

        print("end of epoch")
    return 0


main()