from typing import List
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn


class ClassificationNetwork(nn.Module):

    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super(ClassificationNetwork, self).__init__()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.gpu = device
        # Structure to define possible classes
        self.actions_classes = np.array([
            [0.0, 0.0, 0.0],  # STRAIGHT
            [0.0, 0.5, 0.0],  # ACCELERATE
            [1.0, 0.0, 0.0],  # RIGHT
            [1.0, 0.0, 0.8],  # RIGHT_BRAKE
            [0.0, 0.0, 0.8],  # BRAKE
            [-1.0, 0.0, 0.8],  # LEFT_BRAKE
            [-1.0, 0.0, 0.0],  # LEFT
            [1.0, 0.5, 0.0],  # RIGHT_ACCEL
            [-1.0, 0.5, 0.0]  # LEFT_ACCEL
        ], dtype=np.float32)
        self.num_classes = self.actions_classes.shape[0]

        # Model Definition
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1),  # RGB input case
            nn.LeakyReLU(negative_slope=0.2),
            # nn.MaxPool2d(kernel_size=2) # 48x48 image size
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.MaxPool2d(kernel_size=2, stride=2) # 12x12 image size
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.MaxPool2d(kernel_size=2, stride=2) # 12x12 image size
        )
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=8 * 22 * 22 + 7, out_features=64),  # 12 * 12 * 3 * 32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),  # 12 * 12 * 3 * 32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(32, self.num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, observation, sensor_data):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        sensor_data:   tuple of size 4 containing 4 different types of sensor data
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        # Unpack tuple data
        speed, abs_sensors, steering, gyroscope = sensor_data
        # Define sequential forwarding model
        out = self.layer1(observation.permute(0, 3, 1, 2))
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = torch.flatten(out)
        # out = out.reshape(out.size(0), -1)
        out = out.reshape(-1, 8 * 22 * 22)
        out = self.dropout(out)
        out = torch.cat((out, speed, abs_sensors, steering, gyroscope), 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        out = self.softmax(out)

        return out

    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are number_of_classes
        different classes, then every action is represented by a
        number_of_classes-dim vector which has exactly one non-zero entry
        (one-hot encoding). That index corresponds to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size number_of_classes
        """
        # Convert actions array to id (value ranging from 0 to number_of_classes
        class_labels: List[Tensor] = []

        for action in actions:
            action = np.array(action)
            # Retrieve action class ID
            action_id = np.where(np.all(self.actions_classes ==
                                        np.array(action), axis=1))
            if len(action_id[0]) == 0:
                action_id = 0
                print("action not recognized")
            else:
                action_id = action_id[0][0]
            # Create one-hot encoding
            label = np.zeros(self.num_classes)
            label[action_id] = 1.0
            # Append class label
            class_labels.append(torch.tensor(label))

        return class_labels

    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        return          (float, float, float)
        """
        # Compute maximum score value
        _, predicted = torch.max(scores.data, 1)
        return self.actions_classes[predicted]


