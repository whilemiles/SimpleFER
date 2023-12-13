"""train.py"""
import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch import optim
import Emo_CNN

PATH_TRAIN = './dataset/train/'
PATH_TEST = './dataset/validate/'
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.01

def train(train_dataset, val_dataset, batch_size, epochs,
          learning_rate, wt_decay, print_cost=True):
    """train"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)
    model = Emo_CNN.EmoCNN()
    model = model.to(device)
    compute_loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)

    for epoch in range(epochs):
        loss = 0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()
        if print_cost:
            print(f"epoch{epoch + 1}: train loss:{loss.item()}")
        if epoch % 10 == 9:
            model.eval()
            acc_train = validate(model, train_dataset, batch_size)
            acc_val = validate(model, val_dataset, batch_size)
            print(f"acc train: {acc_train * 100:0.1f}%")
            print(f"acc val: {acc_val * 100:0.1f}%")
    return model


def validate(model, dataset, batch_size):
    """validate"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader = data.DataLoader(dataset, batch_size, shuffle=True)
    result, total = 0.0, 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        pred = model.forward(images)
        pred_tmp = pred.cuda().data.cpu().numpy()
        pred = np.argmax(pred_tmp, axis=1)
        labels = labels.data.cpu().numpy()
        result += np.sum((pred == labels))
        total += len(images)
    acc = result / total
    return acc


def main():
    """main"""
    transforms_train = transforms.Compose([
            transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor()
    ])
    transforms_vaild = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    data_train = torchvision.datasets.ImageFolder(root=PATH_TRAIN,transform=transforms_train)
    data_vaild = torchvision.datasets.ImageFolder(root=PATH_TEST,transform=transforms_vaild)

    model = train(data_train, data_vaild, batch_size=BATCH_SIZE, epochs=EPOCHS,
                  learning_rate=LEARNING_RATE, wt_decay=0, print_cost=True)
    os.makedirs('./saved', exist_ok=True)
    torch.save(model, './saved/EmoCNN.pt')  # 保存模型

if __name__ == '__main__':
    main()
