import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
import torch
import numpy as np
import PIL
#import imgaug as ia
#from imgaug import augmenters as iaa


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
    result = []
    # Define Transformation
    transformation = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.CenterCrop((122, 122)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.4161, ], [0.1688, ]),
    ])
    for x in frames:
        x = mask_image(x)
        reshaped = transformation(x.to(cpu).permute(2, 0, 1))
        result.append(reshaped.permute(1, 2, 0))
    # result = torch.reshape(torch.cat(result, dim=0), (-1, 96, 96, 1)).to(device)
    return result


def replace_color(image, old_color, new_color):
    # Replace the colors defined bellow
    mask = np.all(image == old_color, axis=2)
    image[mask] = new_color
    return image

def mask_image(frame):
    """ Preprocess the images (states) of the expert dataset before feeding them to agent """
    if type(frame)== list:
        # Reshape list
        frame = torch.reshape(torch.cat(frame, dim=0), (-1, 96, 96, 3))

    new_frame = np.copy(frame.cpu())

    # Paint black over the sum of rewards metadata
    new_frame[:, 85:, :15] = [0.0, 0.0, 0.0]

    # Black bar
    new_frame = replace_color(new_frame, [000., 000., 000.], [120.0, 120.0, 120.0])

    # Road
    road_mask = [102.0, 102.0, 102.0]
    new_frame = replace_color(new_frame, [102., 102., 102.], road_mask)
    new_frame = replace_color(new_frame, [105., 105., 105.], road_mask)
    new_frame = replace_color(new_frame, [107., 107., 107.], road_mask)
    # Curbs
    new_frame = replace_color(new_frame, [255., 000., 000.], road_mask)
    new_frame = replace_color(new_frame, [255., 255., 255.], road_mask)
    # Grass
    grass_mask = [102., 229., 102.]
    new_frame = replace_color(new_frame, [102., 229., 102.], grass_mask)
    new_frame = replace_color(new_frame, [102., 204., 102.], grass_mask)

    # Float RGB represenattion
    #new_frame /= 255.

    return torch.tensor(new_frame)


def image_augmentation(frames):
    result = []

    image_transform = transforms.Compose([
        #ImgAugTransform(),
        lambda x: PIL.Image.fromarray(x),
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.Resize((96, 96)),
        transforms.RandomRotation(20),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4161, ], [0.1688, ]),
    ])

    for x in frames:
        reshape = image_transform(x.detach().numpy().astype(np.uint8))
        result.append(reshape.permute(1, 2, 0))

    return result

