import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger  # W&B
import argparse
import wandb


from models.model_basic.model import TransformerClassifier
from models.model_multi_attention.model import SimplifiedMultiAttentionTransformer

from models.model_main_V3.hybrid_model import HybridTransformerClassifier
from models.model_main_V3.dataloader import HybridDataModule

CONFIG_PATH = r"C:\Users\AGFirass\Documents\GitHub\Transformer-Based-DDoS-Detection\models\model_main_V3\config.yaml"


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_callbacks(config):
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['training']['checkpoint_dir'],
        filename='{epoch}-{val_acc:.4f}-{val_loss:.4f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=False
    )
    callbacks.append(checkpoint_callback)

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=config['training']['early_stopping_patience'],
        verbose=True,
        min_delta=0.001
    )
    callbacks.append(early_stop_callback)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    return callbacks


def train_hybrid_model(implicit_ckpt_path: str = None,
                       explicit_ckpt_path: str = None,
                       resume_from_checkpoint: str = None):


    config = load_config(CONFIG_PATH)
    print("Configuration loaded successfully")

    data_path = config['data']['data_path']
    print(f"Using data from: {data_path}")

    pl.seed_everything(config['training']['random_seed'])

    print("Setting up data module...")
    data_module = HybridDataModule(
        data_path=data_path,
        feature_groups=config['feature_groups'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_state=config['training']['random_seed']
    )

    data_module.setup()
    num_classes = data_module.get_num_classes()
    num_features = data_module.get_num_features()

    print(f"Dataset info:")
    print(f"   - Number of classes: {num_classes}")
    print(f"   - Number of features: {num_features}")

    config['model']['num_classes'] = num_classes
    if 'num_features' not in config['model']:
        config['model']['num_features'] = num_features

    implicit_config_path = r"C:\Users\AGFirass\Documents\GitHub\Transformer-Based-DDoS-Detection\models\model_basic\config\config.yaml"
    explicit_config_path = r"C:\Users\AGFirass\Documents\GitHub\Transformer-Based-DDoS-Detection\models\model_multi_attention\config\config.yaml"

    print("Loading sub-model configs...")
    with open(implicit_config_path, "r") as f:
        implicit_config = yaml.safe_load(f)

    with open(explicit_config_path, "r") as f:
        explicit_config = yaml.safe_load(f)

    implicit_config['model']['num_classes'] = num_classes
    explicit_config['model']['num_classes'] = num_classes

    print("Initializing hybrid model...")
    model = HybridTransformerClassifier(
        implicit_model_class=TransformerClassifier,
        explicit_model_class=SimplifiedMultiAttentionTransformer,
        implicit_config=implicit_config,
        explicit_config=explicit_config,
        fusion_config=config['fusion']
    )

    print(f"Hybrid model created with {model.fusion_method} fusion")
    print(f"   - Implicit weight: {model.implicit_weight}")
    print(f"   - Explicit weight: {model.explicit_weight}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - Total Trainable Parameters: {total_params:,}")

    if implicit_ckpt_path or explicit_ckpt_path:
        print("Loading pretrained weights...")
        model.load_pretrained_weights(implicit_ckpt_path, explicit_ckpt_path)

    callbacks = setup_callbacks(config)

    logger = WandbLogger(
        name="hybrid-transformer-run",
        save_dir=config['training']['log_dir'],
        config=config,
        log_model="all",
        entity=None
    )

    logger.experiment.config.update({
        "implicit_config": implicit_config,
        "explicit_config": explicit_config
    })

    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        precision=config['training'].get('precision', 32),
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=50
    )

    print("\n" + "=" * 50)
    print("** STARTING HYBRID MODEL TRAINING **")
    print("=" * 50)
    print(f"   - Max epochs: {config['training']['epochs']}")
    print(f"   - Batch size: {config['training']['batch_size']}")
    print(f"   - Learning rate: {config['fusion']['learning_rate']}")
    print(f"   - Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"   - W&B Run: {logger.name} (Version: {logger.version})")
    print("=" * 50 + "\n")

    try:
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=resume_from_checkpoint
        )
        print("=" * 50)
        print(" TRAINING COMPLETED SUCCESSFULLY ")
        print("=" * 50)

    except Exception as e:
        print("\n" + "!" * 50)
        print(f" TRAINING FAILED WITH EXCEPTION: {e}")
        print("!" * 50)

    finally:
        if logger:
            print("Syncing final logs and artifacts to W&B...")
            wandb.finish()
            print("W&B sync complete.")


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid Transformer Model')

    parser.add_argument('--implicit_ckpt', type=str, default=None,
                        help='Path to pretrained implicit model checkpoint')
    parser.add_argument('--explicit_ckpt', type=str, default=None,
                        help='Path to pretrained explicit model checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to resume hybrid model training from checkpoint')

    args = parser.parse_args()

    train_hybrid_model(
        implicit_ckpt_path=args.implicit_ckpt,
        explicit_ckpt_path=args.explicit_ckpt,
        resume_from_checkpoint=args.resume
    )


if __name__ == "__main__":
    main()