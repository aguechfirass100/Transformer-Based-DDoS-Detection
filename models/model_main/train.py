import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml

from dataloader import create_data_loaders
from model import DDoSTransformerModel


class DDoSModel(pl.LightningModule):
    def __init__(self, config, flow_dim, packet_dim, num_classes, class_weights):
        super().__init__()
        self.config = config
        self.model = DDoSTransformerModel(config, flow_dim, packet_dim)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.num_classes = num_classes

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        logits, _ = self(batch)
        # Use first flow's label as group label
        labels = batch['labels'][:, 0]
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, _ = self(batch)
        labels = batch['labels'][:, 0]  # Group label
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['num_epochs']
        )

        return [optimizer], [scheduler]


class DDoSDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_loader, self.val_loader, self.label_encoder, _ = create_data_loaders(self.config)


        # Get class weights
        class_counts = torch.zeros(len(self.label_encoder.classes_))
        for batch in self.train_loader:
            labels = batch['labels'][:, 0]
            for label in labels:
                class_counts[label] += 1
        self.class_weights = len(class_counts) * class_counts.sum() / (class_counts + 1e-6)
        self.class_weights = torch.clamp(self.class_weights, max=10.0)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize WandB
    wandb_logger = WandbLogger(
        project=config['logging']['experiment_name'],
        log_model='all',
        save_dir=config['logging']['save_dir']
    )

    # Setup data and model
    data_module = DDoSDataModule(config)
    data_module.setup()

    sample_batch = next(iter(data_module.train_dataloader()))
    model = DDoSModel(
        config,
        flow_dim=sample_batch['flow_features'].shape[-1],
        packet_dim=sample_batch['packets'].shape[-1],
        num_classes=config['data']['num_classes'],
        class_weights=data_module.class_weights
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['logging']['save_dir'],
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta'],
        mode='min'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=config['logging']['log_interval']
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()