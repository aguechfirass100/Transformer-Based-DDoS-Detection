import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from flow_dataset import FlowDataModule
from model import TransformerClassifier

def main():
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    pl.seed_everything(config.get('seed', 42))

    # Initialize DataModule
    data_cfg = config['data']
    datamodule = FlowDataModule(
        data_file=data_cfg['data_file'],
        label_column=data_cfg['label_column'],
        train_frac=data_cfg.get('train_fraction', 0.8),
        val_frac=data_cfg.get('val_fraction', 0.1),
        test_frac=data_cfg.get('test_fraction', 0.1),
        batch_size=config['training']['batch_size']
    )
    print("###### - DataModule initialized")

    # Initialize model
    model = TransformerClassifier(config)
    print("###### - Model initialized")

    # Setup checkpoint callback: saves best model by val_loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.get('output_dir', '.'), 'checkpoints'),
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        filename='best-checkpoint-{epoch:02d}-{val_loss:.4f}',
        save_weights_only=True,
        verbose=True
    )

    early_cfg = config['early_stopping']
    early_stop_callback = EarlyStopping(
        monitor=early_cfg['monitor'],
        patience=early_cfg['patience'],
        verbose=True,
        mode=early_cfg['mode']
    )

    # Setup logger (TensorBoard)
    tb_logger = TensorBoardLogger(
        save_dir=config.get('output_dir', '.'),
        name='tensorboard_logs'
    )

    wandb_logger = WandbLogger(project='DDoS Detection', log_model=True)

    # Trainer with callbacks and logger
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator="gpu" if config['training'].get('gpus', 0) > 0 else "cpu",
        devices=config['training'].get('gpus', 0) or 1,
        default_root_dir=config.get('output_dir', '.'),
        callbacks=[checkpoint_callback, early_stop_callback],
        # logger= tb_logger
        logger = [tb_logger, wandb_logger]
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
