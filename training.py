import torch
import torch.nn as nn
import torchvision
import random
import time
import utils
from network import ClassificationNetwork
from imitations import load_imitations
import torchvision.transforms as transforms


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

    batches = [batch for batch in zip(observations,
                                      infer_action.action_to_multilabel(actions))]

    nr_epochs = 100
    batch_size = 64
    number_of_classes = infer_action.num_classes  # needs to be changed
    start_time = time.time()
    prev_loss = 100000

    for epoch in range(nr_epochs):
        random.shuffle(batches)

        total_loss = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):
            batch_in.append(batch[0].to(device))
            batch_gt.append(batch[1].to(device))

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                batch_in = torch.reshape(torch.cat(batch_in, dim=0), (-1, 96, 96, 3))
                sensor = utils.extract_sensor_values(batch_in, batch_size)
                batch_in = utils.preprocess_image(batch_in)
                batch_in = torch.reshape(torch.cat(batch_in, dim=0), (-1, 96, 96, 1))
                batch_gt = torch.reshape(torch.cat(batch_gt, dim=0),
                                         (-1, number_of_classes))

                batch_out = infer_action(batch_in, sensor)
                loss = binary_cross_entropy_loss(batch_out, batch_gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss

                batch_in = []
                batch_gt = []

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))
        if total_loss < prev_loss:
            prev_loss = total_loss
            torch.save(infer_action, trained_network_file)
            print('saved_model')


def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
    batch_out:      torch.Tensor of size (batch_size, number_of_classes)
    batch_gt:       torch.Tensor of size (batch_size, number_of_classes)
    return          float
    """
    """
    # Define loss function
    loss = torch.nn.CrossEntropyLoss()
    # Compute labels
    _, labels = batch_gt.max(dim=1)
    # Compute loss function
    out = loss(batch_out, labels)
    #print(out)
    return out
    """
    # _, batch_gt = batch_gt.max(dim=1)
    epsilon = 0.0001
    loss = batch_gt * torch.log(batch_out + epsilon) + \
           (1 - batch_gt) * torch.log(1 - batch_out + epsilon)
    loss = -torch.mean(torch.sum(loss, dim=1), dim=0)

    return loss

def binary_cross_entropy_loss(batch_out, batch_in):

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(batch_out, batch_in)
    return loss
