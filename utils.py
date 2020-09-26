import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt


def extract_sensor_values(observation, batch_size):
    """
    observation:    python list of batch_size many torch.Tensors of size
                    (96, 96, 3)
    batch_size:     int
    return          torch.Tensors of size (batch_size, 1),
                    torch.Tensors of size (batch_size, 4),
                    torch.Tensors of size (batch_size, 1),
                    torch.Tensors of size (batch_size, 1)
    """
    # print(observation.shape)
    if observation.shape[0] < batch_size:
        batch_size = observation.shape[0]
    speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
    speed = speed_crop.sum(dim=1, keepdim=True) / 255
    abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
    abs_sensors = abs_crop.sum(dim=1) / 255
    steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
    steering = steer_crop.sum(dim=1, keepdim=True)
    gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
    gyroscope = gyro_crop.sum(dim=1, keepdim=True)
    return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope


def preprocess_image(frames):
    """
    Preprocess a list of tensor images frame by normalizing image, converting
    into grayscale, and transforming into tensor
    :param frames: pytorch tensor of size [batch, size, x,y,3] (RGB) image
    :return: torch tensors of size [batch_size, x, y, 1]
    """
    cpu = torch.device('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = []
    for x in frames:
        reshape = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.CenterCrop((122, 122)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.4161, ], [0.1688, ]),
        ])(x.to(cpu).permute(2, 0, 1))
        result.append(reshape.permute(1, 2, 0))
    # result = torch.reshape(torch.cat(result, dim=0), (-1, 96, 96, 1)).to(device)
    return result

def plot_history(accuracy, loss, epochs):
    """
    Description
    ----------
    Generates plot accuracy and loss history for training and validation
    Parameters
    ----------
        :param train: tuple
        tuple containing (loss_train, acc_train) history
        :param test: tuple
        tuple containing (loss_test, acc_test) history
        :param epochs : int
        Integer describing number of epochs for which the model was ran trhough.

    """
    # Define plots
    plt.figure(figsize=(18, 5), dpi=80, facecolor='w', edgecolor='k')
    # plt.rcParams['figure.figsize'] = [15, 5]
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    # Plot Accuracy
    ax1.plot(accuracy, label='Training Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    # ax1.set_ylim([0.8, 1])
    ax1.legend(loc='lower right')
    ax1.grid(True)
    ax1.set_title('Accuracy Over ' + str(epochs) + ' Epochs')

    # Plot Loss
    ax2.plot(loss, label='Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    # ax2.set_ylim([0, 3])
    ax2.legend(loc='upper right')
    ax2.grid(True)
    ax2.set_title('Loss Over ' + str(epochs) + ' Epochs')
    plt.savefig('History.png')
    plt.show()
