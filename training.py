import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import random
import time
import utils
from network import ClassificationNetwork
from imitations import load_imitations
from sklearn.metrics import accuracy_score


def train(data_folder, trained_network_file):
    """
    Function for training the network.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_action = ClassificationNetwork().to(device)
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=2e-3)
    observations, actions = load_imitations(data_folder)
    observations = [torch.Tensor(observation) for observation in observations]
    actions = [torch.Tensor(action) for action in actions]
    # Preprocess data
    # Augment dataset

    observations_aug = utils.image_augmentation(observations)
    actions_aug = actions
    # Append new augmented dataset
    observations = observations_aug + observations
    actions = actions + actions_aug

    # Generate batches
    batches = [batch for batch in zip(observations,
                                      infer_action.actions_to_classes(actions))]

    nr_epochs = 100
    batch_size = 64
    number_of_classes = infer_action.num_classes  # needs to be changed
    start_time = time.time()
    prev_loss = 100000
    loss_train_hist = []
    acc_train_hist = []

    for epoch in range(nr_epochs):
        random.shuffle(batches)

        loss_train = 0
        batch_in = []
        batch_gt = []
        # Train
        for batch_idx, batch in enumerate(batches):
            batch_in.append(batch[0].to(device))
            batch_gt.append(batch[1].to(device))

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                batch_in = torch.reshape(torch.cat(batch_in, dim=0), (-1, 96, 96, 3))
                batch_gt = torch.reshape(torch.cat(batch_gt, dim=0),
                                         (-1, number_of_classes))
                batch_out = infer_action(batch_in)
                loss = cross_entropy_loss(batch_out, batch_gt)
                # Loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train += loss
                # Accuracy
                scores_predicted = F.softmax(batch_out, dim=1)
                _, y_predicted = scores_predicted.max(dim=1)
                _, y_truth = batch_gt.max(dim=1)
                acc_train = accuracy_score(y_truth, y_predicted)
                batch_in = []
                batch_gt = []

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f\t accuracy: %.6f\tETA: +%fs" % (
            epoch + 1, loss_train, acc_train, time_left))
        loss_train_hist.append(loss_train)
        acc_train_hist.append(acc_train)
        if loss_train < prev_loss:
            prev_loss = loss_train
            torch.save(infer_action, trained_network_file)
            print('saved_model')
    utils.plot_history(acc_train_hist, loss_train_hist, nr_epochs)


def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
    batch_out:      torch.Tensor of size (batch_size, number_of_classes)
    batch_gt:       torch.Tensor of size (batch_size, number_of_classes)
    return          float
    """
    weights = torch.tensor([0.8, 0.8, 1, 1, 1, 1, 1, 1, 1]).to(\
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Define loss function
    loss = torch.nn.CrossEntropyLoss(weight = weights)
    # Compute labels
    _, labels = batch_gt.max(dim=1)
    # Compute loss function
    out = loss(batch_out, labels)
    # print(out)
    return out


def binary_cross_entropy_loss(batch_out, batch_in):
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(batch_out, batch_in)
    return loss
