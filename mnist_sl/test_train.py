import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from torch.utils.data import DataLoader
import os
import numpy as np
from models import CNN, CNNDropout, FeedForwardNN
from utils import train, test, test_train_split

# refactor to train models as needed
def main():
    models = [CNN(), CNNDropout(0.3), FeedForwardNN(2, [128, 128])]
    PATHS = ['./test_train_models/cnn.pth', './test_train_models/cnn_dropout.pth', './test_train_models/ffnn.pth']
    epochs = 5

    trainloader, testloader = test_train_split(16) # batch size
    for i, model in enumerate(models):
        # load model if exists
        if os.path.exists(PATHS[i]):
            model.load_state_dict(torch.load(PATHS[i]))
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        train(epochs, model, trainloader, optim, criterion, PATHS[i])
        accuracy = test(model, testloader)
        print(accuracy)
        

if __name__ == '__main__':
    main()


# Test-Train Split

# CNN using Adam Optimizer and standard layers (2 conv, 1 pool, 1 fully-connected)
# After 1 epoch:
# loss: 0.000
# accuracy: 89.7% -> definetely overfitting

# CNN using Adam Optimizer and standard layers (2 conv, 1 pool, 1 fully-connected, 1 dropout(p=0.4))
# After 6 epochs:
# loss: 0.020
# accuracy: 86.9%


# Feed-Forward NN
# with 2 layers, [128, 128], test accuracies: 0.587, 0.587 avg loss: about 1.00 over 10 total epochs (5 + 5)
# with 3 layers, [128, 192, 128], test accuracies: 0.352,0.337 avg loss: about 1.7 over 10 total epochs (5 + 5)