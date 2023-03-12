import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy


class AlexNet(nn.Module):
    
    def __init__(self, out_dim):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 192, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)
        )
        
        self.classifier=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*2*2, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, out_dim)
        )
        
        
    def forward(self, x):
        x = self.feature_extractor(x)
        feat = x.reshape(x.shape[0],-1)
        x = self.classifier(feat)

        return x, feat
    
    
class AlexNetLit(pl.LightningModule):
    
    def __init__(self, out_dim, lr):
        super().__init__()
        
        self.model = AlexNet(out_dim)
        
        self.loss = nn.CrossEntropyLoss()        
        self.save_hyperparameters()
        
        self.train_accuracy = Accuracy(task="multiclass", num_classes=out_dim)
        self.valid_accuracy = Accuracy(task="multiclass", num_classes=out_dim)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=out_dim)
        
        
    def forward(self, x):
        x = self.model.feature_extractor(x)
        feat = x.reshape(x.shape[0],-1)
        x = self.model.classifier(feat)

        return x, feat
    
    
    def configure_optimizers(self):
        print(self.hparams['lr'])
        return torch.optim.Adam(self.parameters(), lr = self.hparams['lr'])
    
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        y_pred, feat = self(images)
        preds = torch.argmax(y_pred, 1)
        
        loss = self.loss(y_pred, labels)
        acc = self.train_accuracy(y_pred, labels)
        
        self.log('TRAIN LOSS', loss, on_epoch = True)
        self.log('TRAIN ACCURACY', acc, on_epoch = True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        y_pred, feat = self(images)
        preds = torch.argmax(y_pred, 1)
        
        loss = self.loss(y_pred, labels)
        acc = self.valid_accuracy(y_pred, labels)
        
        self.log('VALIDATION LOSS', loss, on_step = False, on_epoch= True)
        self.log('VALIDATION ACCURACY', acc, on_step = False, on_epoch = True)
        
        return loss
    
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        y_pred, feat = self(images)
        preds = torch.argmax(y_pred, 1)
        
        loss = self.loss(y_pred, labels)
        acc = self.test_accuracy(y_pred, labels)
        
        self.log('TEST LOSS', loss, on_step = False, on_epoch= True)
        self.log('TEST ACCURACY', acc, on_step = False, on_epoch = True)
        
        return loss
    
    

    