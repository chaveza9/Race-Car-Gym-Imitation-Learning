import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
import torch


def bbc(frames):

    reshape = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.CenterCrop((122, 122)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.4161, ], [0.1688, ]),
    ])

    result = [reshape()(x) for x in frames]
    result = torch.stack(result)

    return result
