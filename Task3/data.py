import torch
import torchvision
from torch.utils.data import DataLoader
from visualization import Visualization
from PIL import Image

BATCH_SIZE = 16

data_set = torchvision.datasets.FashionMNIST(
    root="./data/",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)
data_sampler = DataLoader(data_set, batch_size=1, shuffle=True)

view = Visualization(data_sampler, data_set.classes)


def transform_all():
    images = []
    for img_raw in data_set.data:
        img_array = Image.fromarray(img_raw.numpy(), mode='L')
        img_tensor = data_set.transform(img_array)
        images.append(img_tensor)

    return torch.stack(images)


if __name__ == "__main__":
    view.preview()
