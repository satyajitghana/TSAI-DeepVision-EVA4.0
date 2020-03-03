from sodium.base import BaseDataLoader

from torchvision import datasets
from torch.utils.data import DataLoader


class MNISTDataLoader(BaseDataLoader):

    def __init__(self, transforms, data_dir, batch_size, shuffle, nworkers, train=True):
        self.data_dir = data_dir

        self.train_loader = datasets.MNIST(
            self.data_dir,
            train=train,
            download=True,
            transform=transforms.build_transforms(train=True)
        )

        self.test_loader = datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=transforms.build_transforms(train=False)
        )

        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': nworkers
        }
        super().__init__(self.train_loader, shuffle=shuffle, **self.init_kwargs)

    def test_split(self):
        return DataLoader(self.test_loader, **self.init_kwargs)


class CIFAR10DataLoader(BaseDataLoader):

    def __init__(self, transforms, data_dir, batch_size, shuffle, nworkers, train=True):
        self.data_dir = data_dir

        self.train_loader = datasets.CIFAR10(
            self.data_dir,
            train=train,
            download=True,
            transform=transforms.build_transforms(train=True)
        )

        self.test_loader = datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=transforms.build_transforms(train=False)
        )

        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': nworkers
        }
        super().__init__(self.train_loader, shuffle=shuffle, **self.init_kwargs)

    def test_split(self):
        return DataLoader(self.test_loader, **self.init_kwargs)
