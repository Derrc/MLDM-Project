import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from dataset import SLDataset
from torch.utils.data import DataLoader
import os
# from cnn_model import train, test, save, load

class FeedForwardNN(nn.Module):
    """
    layers is an integer representing number of hidden layers in network
    layer_nodes is a list of n integers, where the ith element is the number of nodes in layer i
    """
    def __init__(self, layers, node_count):
        super(FeedForwardNN, self).__init__()
        self.act_func = nn.ReLU()
        # create list of layers (input + hidden layers + out)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(784, node_count[0]))
        for i in range(layers - 1):
            self.layers.append(nn.Linear(node_count[i], node_count[i+1]))
        self.layers.append(nn.Linear(node_count[-1], 26))
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        out = x
        for layer in self.layers:
            out = self.act_func(layer(out))
        return out

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

    PATH = './models/model_ffnn.pth'
    model = FeedForwardNN(3, [128, 192, 128])
    # input -> relu -> relu -> output
    if os.path.exists(PATH):
        load(model, PATH)

    # train, number of iterations through dataset
    train(5)

    test()

    # save model
    save(model, PATH)

    # with 2 layers, [128, 128], test accuracies: 0.587, 0.587 avg loss: about 1.00 over 10 total epochs (5 + 5)
    # with 3 layers, [128, 192, 128], test accuracies: 0.352,0.337 avg loss: about 1.7 over 10 total epochs (5 + 5)
