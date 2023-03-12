import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, random_split


class CIFAR10Data(pl.LightningDataModule):
    
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        self.transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))])
        
        
    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            cifar = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar, [45000, 5000])
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)


    def train_dataloader(self):
        cifar_train = DataLoader(self.cifar_train, batch_size=self.batch_size)
        return cifar_train


    def val_dataloader(self):
        cifar_val = DataLoader(self.cifar_val, batch_size=10 * self.batch_size)
        return cifar_val


    def test_dataloader(self):
        cifar_test = DataLoader(self.cifar_test, batch_size=10 * self.batch_size)
        return cifar_test