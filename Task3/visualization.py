import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


class Visualization(object):
    def __init__(self, data_loader, labels_map):
        self.sampler = data_loader
        self.labels_map = labels_map

        self.samples = self._sample()
        self.loss_records = []
        self.sample_decoded = []

    def preview(self):
        cols, rows = 5, 2
        figure = plt.figure(figsize=(5, 3))

        for i in range(cols * rows):
            figure.add_subplot(rows, cols, i + 1)
            plt.title(self.labels_map[i])
            plt.axis("off")
            plt.imshow(self.samples[i].squeeze(), cmap="gray")

        plt.show()

    def _sample(self):
        sampler = iter(self.sampler)
        samples = [None for _ in range(len(self.labels_map))]

        while None in samples:
            sample_image, sample_label = next(sampler)
            if samples[sample_label] is None:
                samples[sample_label] = sample_image[0]

        # noinspection PyTypeChecker
        return torch.stack(samples)

    def comparison(self, net: torch.nn.Module):
        net.eval()
        with torch.no_grad():
            _encoded, decoded = net(self.samples)

        cols, rows = 5, 2
        figure = plt.figure(figsize=(5, 4))

        for i in range(cols * rows):
            figure.add_subplot(rows, cols, i + 1)
            plt.title(self.labels_map[i])
            plt.axis("off")
            plt.imshow(make_grid([self.samples[i], decoded[i]], nrow=1, pad_value=1)[0], cmap="gray")

        plt.show()

    @staticmethod
    def projection(data, labels):
        cols, rows = 2, len(data) / 2
        figure = plt.figure(figsize=(8, 4 * rows))

        for i, (method, dots) in enumerate(data.items()):
            figure.add_subplot(rows, cols, i + 1)
            plt.title(method)
            plt.scatter(*dots.T, s=2, c=labels, cmap="Spectral")

        plt.show()
