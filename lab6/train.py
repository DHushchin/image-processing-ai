from model import AlexNetLit
from dataloader import CIFAR10Data

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
import torch


def main(): 
    lr = 0.001
    batch_size = 64
    out_dim = 10
    data_dir = 'data'
    
    cifar = CIFAR10Data(data_dir, batch_size)
    
    model = AlexNetLit(out_dim, lr)
    
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=10,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        logger=CSVLogger(save_dir="logs/"),

    )

    trainer.fit(model, cifar)
    trainer.test(model, datamodule=cifar)
    trainer.save_checkpoint('weights/alexnet.ckpt')
    
    
if __name__ == "__main__":
    main()