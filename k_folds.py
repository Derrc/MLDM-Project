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
    PATHS = [
        "./models_k_folds/cnn.pth",
        "./models_k_folds/cnn_dropout.pth",
        "./models_k_folds/ffnn.pth",
    ]
    for i, model in enumerate(models):
        mean_accuracy = k_folds_cross_validation(
            4, 16, model, PATHS[i]
        )
        print(mean_accuracy)


if __name__ == "__main__":
    main()


# K-Folds Cross Validation

# CNN using Adam Optimizer and standard layers (2 conv, 1 pool, 1 fully-connected)
# Over 4 epochs using 4-Folds Cross-Validation
# average loss -> 0.000
# mean accuracy over validation sets -> 1.000
# accuracy on given test set = 100%

# CNN using Adam Optimizer and standard layers (2 conv, 1 pool, 1 fully-connected, 1 dropout(p=?))
# Over 4 epochs using 4-Fold Cross-Validation
# average loss over each epoch -> 0.35, 0.05, 0.03, 0.02
# mean accuracy over validation sets -> .990
# accuracy on given test set = 98.69%

# FFNN using Adam Optimizer with 2 hidden layers, each 128 nodes tall
# Over 4 epochs using 4-Fold Cross-Validation
# average loss over each epoch -> 2.1, 1.7, 1.7, 1.63
# mean accuracy over validation sets -> 0.541
# accuracy on given test set = 52%

# I'm going to train this one for more epochs... and see what happens
# nothing, nothing happened. It made no significant progress after 52% accuracy
