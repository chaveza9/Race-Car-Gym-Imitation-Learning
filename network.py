from typing import List
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationNetwork(nn.Module):

    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super(ClassificationNetwork, self).__init__()
        gpu = torch.device('cuda')
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
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),  # RGB input case
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=4))  # 24x24 image size
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 12x12 image size
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Sequential(
            nn.Linear(5408, 128),   #12 * 12 * 3 * 32
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, self.num_classes),

        )

    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        # Define sequential forwarding model
        out = self.layer1(observation.permute(0, 3, 2, 1))
        out = self.dropout(out)
        out = self.layer2(out)
        #out = torch.flatten(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
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
            else:
                action_id = action_id[0][0]
            # Create one-hot encoding
            label = np.zeros(self.num_classes)
            label[action_id] = 1.0
            # Append class label
            class_labels.append(torch.tensor(label))

        return class_labels

    def __check_invalid_actions__(self, action):
        """
        Checks if there is any forbidden action in the expert dataset
        actions:        python list of N torch.Tensors of size 3
        returns:
        Flag:           flag if invalid action detected
        substitute_action    substitute to class action
        """
        invalid_actions = np.array([
            [0.0, 0.5, 0.8],  # ACCEL_BRAKE
            [1.0, 0.5, 0.8],  # RIGHT_ACCEL_BRAKE
            [-1.0, 0.5, 0.8],  # LEFT_ACCEL_BRAKE
            [1.0, 0.5, 0.0],  # RIGHT_ACCEL
            [-1.0, 0.5, 0.0],  # LEFT_ACCEL
        ])
        inv_class = np.where(np.all(invalid_actions == np.array(action), axis=1))[0][0]

        if len(inv_class) > 0:
            flag = 1

        return flag

    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        return          (float, float, float)
        """

        label_indx = 0
        # Compute maximum score value
        _, predicted = torch.max(scores.data, 1)
        # Retrieve class index
        #loc = np.where(np.array(scores) == 1)
        # Check if index has been located
        #if len(loc[0]) > 0:
        #    label_indx = loc[0][0]
        # Return action map
        return self.actions_classes[predicted]

    def extract_sensor_values(self, observation, batch_size):
        """
        observation:    python list of batch_size many torch.Tensors of size
                        (96, 96, 3)
        batch_size:     int
        return          torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 4),
                        torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 1)
        """
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope


"""
if __name__ == "__main__":
    classification = ClassificationNetwork()

    # Extract actions and observations
    from imitations import load_imitations

    observations, actions = load_imitations('./data/teacher')
    actions = [torch.Tensor(action) for action in actions]

    classes = classification.actions_to_classes(actions)

    a = classification.scores_to_action(classes[0])
"""
