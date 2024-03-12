import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from ResNet18 import ResNet18
from torch.cuda.amp import GradScaler, autocast
from Utils import *
from Constants import *

best_acc = 0.0
def test(model, epoch, dataloader, criterion, writer=None, NCrop=True):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
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
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = running_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    if writer:
        writer.add_scalar('Loss/test', avg_loss, epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)
    return accuracy

def train(model, train_loader, test_loader, optimizer, criterion, scheduler, num_epochs=5, start_epoch=0, writer=None, NCrop=True, scaler = GradScaler()):
    global best_acc
    for epoch in range(start_epoch, num_epochs):
        print("")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # org_images, org_labels = images.clone(), labels.clone()
            with autocast():
                bs, ncrops, c, h, w = images.shape
                images = images.view(-1, c, h, w)
                labels = torch.repeat_interleave(labels, repeats=ncrops, dim=0)
                images, labels_a, labels_b, lam = mixup_data(images, labels, 0.2)
                
                outputs = model(images)
                
                soft_labels_a = smooth_one_hot(labels_a, classes=7, smoothing=0.2)
                soft_labels_b = smooth_one_hot(labels_b, classes=7, smoothing=0.2)
                loss = mixup_criterion(criterion, outputs, soft_labels_a, soft_labels_b, lam)
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        
        print(f'Train Loss: {avg_loss:.4f}')
        if writer:
            writer.add_scalar('Loss/train', avg_loss, epoch)

        scheduler.step(avg_loss)

        acc = test(model, epoch, test_loader, criterion, writer)
        if(acc > best_acc):
            best_acc = acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
            }, SAVED_MODEL_DIR + "_best")
        
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
        }, SAVED_MODEL_DIR)
        
    print("Best acc:" + str(best_acc))        

        
mu, st = 0, 255

transforms_train = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
    transforms.RandomApply([transforms.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
    transforms.RandomApply(
        [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
    transforms.FiveCrop(40),
    transforms.Lambda(lambda crops: torch.stack(
        [transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda tensors: torch.stack(
        [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
    transforms.Lambda(lambda tensors: torch.stack(
        [transforms.RandomErasing()(t) for t in tensors])),
])

transforms_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.TenCrop(40),
    transforms.Lambda(lambda crops: torch.stack(
        [transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda tensors: torch.stack(
        [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
])

random_seed(5050)

data_train = torchvision.datasets.ImageFolder(root=PATH_TRAIN, transform=transforms_train)
data_test = torchvision.datasets.ImageFolder(root=PATH_TEST, transform=transforms_test)
train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True, weight_decay=WEIGHT_DECAY)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

start_epoch = 0
try:
    checkpoint = torch.load(SAVED_MODEL_DIR)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print("Checkpoint loaded, resuming training from epoch", start_epoch)
except FileNotFoundError:
    print("No checkpoint found, starting training from scratch.")

writer = SummaryWriter(TENSORBOARD_DIR)
train(model, train_loader, test_loader, optimizer, criterion, scheduler, EPOCHS, start_epoch=start_epoch, writer=writer)
writer.close()
