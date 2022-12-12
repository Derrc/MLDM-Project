import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from dataset import SLDataset
from torch.utils.data import DataLoader
import os

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # conv -> conv -> pool -> fc -> output(fc)
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2,2)
        # 64 * 12 * 12
        self.fc1 = nn.Linear(9216, 128)
        self.output = nn.Linear(128, 26)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1) # keep batch size, flatten every dim after
        x = F.relu(self.fc1(x))
        x = self.output(x)

        return x

# train function
def train(epochs):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3) # learning rate
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(epochs):
        iterations = 100
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            images, labels = data

            # prediction, forward pass
            labels_pred = model(images)

            # calculate loss
            loss = criterion(labels_pred, labels)

            # compute gradients, backward pass
            loss.backward()

            # update gradients
            optim.step()

            optim.zero_grad()

            running_loss += loss
            if i % iterations == iterations-1:
                print(f'[{e+1}, {i+1}] loss = {running_loss / iterations:.3f}')
                running_loss = 0.0

        # save after each epoch
        save(model, PATH)

    
# test function
def test():
    model.eval()
    total = 0
    correct = 0
    for images, labels in testloader:
        labels_pred = model(images)
        # apply softmax and argmax to get index of class
        labels_pred = torch.argmax(F.softmax(labels_pred, dim=1), dim=1)
        correct += torch.sum(labels_pred == labels).item()
        total += labels.size(0)

    print(f'Accuracy over test set: {correct / total:.3f}')


# save model
def save(model, path):
    torch.save(model.state_dict(), path)

# load model
def load(model, path):
    model.load_state_dict(torch.load(path))



if __name__ == '__main__':
    # toTensor -> [0,1], normalize -> [-1, 1]
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                        torchvision.transforms.Normalize(0.5, 0.5)])
    train_data = SLDataset(transform=transforms, train=True)
    test_data = SLDataset(transform=transforms, train=False)

    trainloader = DataLoader(train_data, batch_size=16, shuffle=True)
    testloader = DataLoader(test_data, batch_size=16, shuffle=False)

    PATH = './models/model_cnn.pth'
    model = CNN()
    if os.path.exists(PATH):
        load(model, PATH)

    # train, number of iterations through dataset
    # train(5)

    test()


# Test-Train Split

# CNN using Adam Optimizer and standard layers (2 conv, 1 pool, 1 fully-connected)
# After 1 epoch:
# loss: 0.000
# accuracy: 89.7% -> definetely overfitting

# CNN using Adam Optimizer and standard layers (2 conv, 1 pool, 1 fully-connected, 1 dropout(p=0.4))
# After 6 epochs:
# loss: 0.020
# accuracy: 86.9%