import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from ResNet18 import ResNet18

from Utils import *
from Constants import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def test(model, dataloader, criterion, NCrop=True):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if NCrop:
                bs, ncrops, c, h, w = inputs.shape
                inputs = inputs.view(-1, c, h, w)
                outputs = model(inputs)
                outputs = outputs.view(bs, ncrops, -1)
                outputs = torch.sum(outputs, dim=1) / ncrops
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = running_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    return accuracy, all_preds, all_labels

model = ResNet18().to(device)
checkpoint = torch.load(SAVED_MODEL_DIR + '_best')
# checkpoint = torch.load(SAVED_MODEL_DIR)
model.load_state_dict(checkpoint['model_state_dict'])

mu, st = 0, 255
transforms_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.TenCrop(40),
    transforms.Lambda(lambda crops: torch.stack(
        [transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda tensors: torch.stack(
        [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
])
criterion = nn.CrossEntropyLoss()

data_test = torchvision.datasets.ImageFolder(root=PATH_TEST, transform=transforms_test)
test_loader = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)

accuracy, all_preds, all_labels = test(model, test_loader, criterion)

conf_matrix = confusion_matrix(all_labels, all_preds)
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix_norm, cmap=plt.cm.Blues, interpolation='nearest')
plt.title('Normalized Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
plt.yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)

for i in range(conf_matrix_norm.shape[0]):
    for j in range(conf_matrix_norm.shape[1]):
        plt.text(j, i, "{:.2f}".format(conf_matrix_norm[i, j]), horizontalalignment="center", color="black")

plt.tight_layout()

plt.show()