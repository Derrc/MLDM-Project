import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from torch.utils.data import DataLoader, random_split, ConcatDataset
import os
import numpy as np
from utils import k_folds_cross_validation
from models import CNN, CNNDropout, FeedForwardNN

# refactor to train models as needed
def main():
    models = [CNN(), CNNDropout(0.3), FeedForwardNN(2, [128, 128])]
    PATHS = ['./models_k_folds/cnn.pth', './models_k_folds/cnn_dropout.pth', './models_k_folds/ffnn.pth']
    for i, model in enumerate(models):
        # load model if exists
        if os.path.exists(PATHS[i]):
            model.load_state_dict(torch.load(PATHS[i]))
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        mean_accuracy = k_folds_cross_validation(4, 16, model, optim, criterion, PATHS[i])
        print(mean_accuracy)

if __name__ == '__main__':
    main()


# K-Folds Cross Validation

# CNN using Adam Optimizer and standard layers (2 conv, 1 pool, 1 fully-connected)
# Over 4 epochs using 4-Folds Cross-Validation
# average loss -> 0.000
# mean accuracy over validation sets -> 0.999
# accuracy on given testset = 100%

# CNN using Adam Optimizer and standard layers (2 conv, 1 pool, 1 fully-connected, 1 dropout(p=?))
