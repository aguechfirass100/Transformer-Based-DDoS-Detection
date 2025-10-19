import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml

from .dataloader import create_data_loaders
from models.model_main_V2.EPT_PPT_CA_model import ExplicitPacketModel



class ExplicitPacketTrainer(pl.LightningModule):
    def __init__(self, config, flow_dim, packet_dim, num_classes, class_weights, feature_info):
        super().__init__()
        self.config = config
        self.model = ExplicitPacketModel(config, flow_dim, packet_dim, feature_info)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.num_classes = num_classes

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        logits, _ = self(batch)
        labels = batch['labels'][:, 0]  # Use first flow's label as group label
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, _ = self(batch)
        labels = batch['labels'][:, 0]
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


class ExplicitPacketDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_loader, self.val_loader, self.label_encoder, self.feature_info = create_data_loaders(self.config)

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
    config_path = r"C:\Users\AGFirass\Documents\GitHub\Transformer-Based-DDoS-Detection\models\model_main_V2\config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("TRAINING EXPLICIT+PACKET MODEL ONLY")
    print("=" * 60)

    # Setup data
    data_module = ExplicitPacketDataModule(config)
    data_module.setup()

    sample_batch = next(iter(data_module.train_dataloader()))

    # Create explicit+packet model
    model = ExplicitPacketTrainer(
        config,
        flow_dim=sample_batch['flow_features'].shape[-1],
        packet_dim=sample_batch['packets'].shape[-1],
        num_classes=config['data']['num_classes'],
        class_weights=data_module.class_weights,
        feature_info=data_module.feature_info
    )

    # Setup logging
    wandb_logger = WandbLogger(
        project=f"{config['logging']['experiment_name']}_explicit_packet",
        log_model='all',
        save_dir=config['logging']['save_dir']
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config['logging']['save_dir']}/explicit_packet",
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

    print(f"\nExplicit+Packet model training completed!")
    print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")

    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    main()