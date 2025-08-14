import torch

torch.set_float32_matmul_precision('medium')

import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from flow_dataset import FlowDataModule
from model import SimplifiedMultiAttentionTransformer as MultiAttentionTransformer


def main():
    # Load config
    config_path = 'C:/Users/AGFirass/Documents/GitHub/Transformer-Based-DDoS-Detection/models/model_multi_attention/config/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    pl.seed_everything(config.get('seed', 42))

    # Initialize DataModule with feature groups
    datamodule = FlowDataModule(
        data_file=config['data']['data_file'],
        feature_groups=config['feature_groups'],
        label_column=config['data']['label_column'],
        train_frac=config['data'].get('train_fraction', 0.7),
        val_frac=config['data'].get('val_fraction', 0.15),
        test_frac=config['data'].get('test_fraction', 0.15),
        batch_size=config['training']['batch_size']
    )

    # Setup data to get actual number of classes
    datamodule.setup()
    actual_num_classes = datamodule.get_num_classes()

    print(f"###### - DataModule initialized with feature groups")
    print(f"###### - Actual number of classes in data: {actual_num_classes}")
    print(f"###### - Class names: {list(datamodule.get_class_names())}")

    # Update config with actual number of classes
    config['model']['num_classes'] = actual_num_classes

    # Initialize model
    model = MultiAttentionTransformer(config)
    print("###### - MultiAttentionTransformer initialized")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"###### - Total parameters: {total_params:,}")
    print(f"###### - Trainable parameters: {trainable_params:,}")

    # Setup checkpoint callback - monitor F1 score for balanced data
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.get('output_dir', '.'), 'checkpoints'),
        monitor='val_f1',  # Changed from val_loss to val_f1
        mode='max',  # Changed from min to max
        save_top_k=3,  # Save top 3 models
        filename='best-checkpoint-{epoch:02d}-{val_f1:.4f}',
        save_weights_only=False,  # Save full model for easier loading
        verbose=True,
        auto_insert_metric_name=False
    )

    # Early stopping callback
    early_cfg = config['early_stopping']
    early_stop_callback = EarlyStopping(
        monitor=early_cfg['monitor'],
        patience=early_cfg['patience'],
        verbose=True,
        mode=early_cfg['mode']
    )

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Setup loggers
    tb_logger = TensorBoardLogger(
        save_dir=config.get('output_dir', '.'),
        name='tensorboard_logs'
    )

    # Initialize wandb with better config tracking
    wandb_logger = WandbLogger(
        project='DDoS Detection - Multi Attention',
        log_model='all',
        save_dir=config.get('output_dir', '.')
    )
    # Log the config to wandb
    wandb_logger.log_hyperparams(config)

    # Trainer configuration with more advanced settings
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator="gpu" if config['training'].get('gpus', 0) > 0 else "cpu",
        devices=config['training'].get('gpus', 0) or 1,
        default_root_dir=config.get('output_dir', '.'),
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=[tb_logger, wandb_logger],
        deterministic=True,
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        precision=16,  # Use mixed precision for faster training
        val_check_interval=1.0,  # Check validation twice per epoch
        log_every_n_steps=50,  # Log more frequently
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Start training
    print("###### - Starting training...")
    trainer.fit(model, datamodule=datamodule)

    print("###### - Training complete")

if __name__ == "__main__":
    main()