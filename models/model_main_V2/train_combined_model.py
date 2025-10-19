import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml
from typing import Optional

from dataloader import create_data_loaders
from combined_model import CombinedDDoSModel, CombinedTrainingStrategy


class CombinedDDoSTrainer(pl.LightningModule):
    def __init__(self, config, flow_dim, packet_dim, num_classes, class_weights,
                 implicit_model_path: Optional[str] = None,
                 explicit_packet_model_path: Optional[str] = None,
                 feature_info: dict = None,
                 training_phase: int = 1):
        super().__init__()
        self.config = config
        self.training_phase = training_phase
        self.automatic_optimization = False  # We'll handle optimization manually

        # Create the combined model
        self.model = CombinedDDoSModel(
            config, flow_dim, packet_dim,
            implicit_model_path, explicit_packet_model_path, feature_info
        )

        # Setup training strategy
        self.training_strategy = CombinedTrainingStrategy(self.model)

        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.num_classes = num_classes

        # Setup training parameters based on phase
        self.setup_training_phase(training_phase)

    def setup_training_phase(self, phase: int):
        """Setup training parameters for different phases"""
        if phase == 1:
            self.trainable_params = self.training_strategy.phase1_train_fusion_only()
            self.learning_rate = 1e-3  # Higher LR for new layers
        elif phase == 2:
            self.trainable_params = self.training_strategy.phase2_finetune_explicit_packet()
            self.learning_rate = 5e-4  # Medium LR for fine-tuning
        elif phase == 3:
            self.trainable_params = self.training_strategy.phase3_finetune_all()
            self.learning_rate = 1e-4  # Lower LR for full fine-tuning
        else:
            raise ValueError(f"Invalid training phase: {phase}")

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        logits, _ = self(batch)
        # Use first flow's label as group label
        labels = batch['labels'][:, 0]
        loss = self.criterion(logits, labels)

        # Manual optimization
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('learning_rate', opt.param_groups[0]['lr'], prog_bar=True)
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
        # Only optimize the trainable parameters for current phase
        optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=self.learning_rate,
            weight_decay=self.config['training']['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['num_epochs']
        )

        return [optimizer], [scheduler]


class CombinedDataModule(pl.LightningDataModule):
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


def train_phase(config, phase: int, implicit_model_path: str = None,
                explicit_packet_model_path: str = None, checkpoint_path: str = None):
    """Train a specific phase of the combined model"""

    print(f"\n{'=' * 50}")
    print(f"TRAINING PHASE {phase}")
    print(f"{'=' * 50}")

    # Setup data
    data_module = CombinedDataModule(config)
    data_module.setup()

    sample_batch = next(iter(data_module.train_dataloader()))

    # Create model for this phase
    model = CombinedDDoSTrainer(
        config,
        flow_dim=sample_batch['flow_features'].shape[-1],
        packet_dim=sample_batch['packets'].shape[-1],
        num_classes=config['data']['num_classes'],
        class_weights=data_module.class_weights,
        implicit_model_path=implicit_model_path,
        explicit_packet_model_path=explicit_packet_model_path,
        feature_info=data_module.feature_info,
        training_phase=phase
    )

    # Load from checkpoint if continuing training
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])

    # Setup logging and callbacks
    wandb_logger = WandbLogger(
        project=f"{config['logging']['experiment_name']}_phase{phase}",
        log_model='all',
        save_dir=config['logging']['save_dir']
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config['logging']['save_dir']}/phase{phase}",
        filename='{epoch}-{val_loss:.2f}-phase{}'.format(phase),
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=config['training']['patience'] // 2 if phase == 1 else config['training']['patience'],
        min_delta=config['training']['min_delta'],
        mode='min'
    )

    # Adjust epochs based on phase
    epochs_per_phase = {
        1: config['training']['num_epochs'] // 3,  # Shorter for fusion-only
        2: config['training']['num_epochs'] // 2,  # Medium for explicit+packet fine-tuning
        3: config['training']['num_epochs']  # Full epochs for complete fine-tuning
    }

    trainer = pl.Trainer(
        max_epochs=epochs_per_phase[phase],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=config['logging']['log_interval']
    )

    trainer.fit(model, datamodule=data_module)

    # Return best checkpoint path
    return checkpoint_callback.best_model_path


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Paths to pre-trained models
    implicit_model_path = r"C:\Users\AGFirass\Documents\GitHub\Transformer-Based-DDoS-Detection\models\model_basic\lightning_logs\checkpoints\best-checkpoint-epoch=22-val_loss=0.0032.ckpt"
    explicit_packet_model_path = None  # Will be trained in this script

    print("Starting Combined Model Training Pipeline")
    print("=" * 60)

    # Step 1: Train Explicit+Packet Model first (if not provided)
    if explicit_packet_model_path is None:
        print("Step 1: Training Explicit+Packet Model...")
        # You need to create a separate training script for this
        # Or train it here before the combined model
        explicit_packet_model_path = train_explicit_packet_model(config)

    # Step 2: Train Combined Model in phases
    print("Step 2: Training Combined Model...")

    # Phase 1: Train fusion layers only
    phase1_checkpoint = train_phase(
        config,
        phase=1,
        implicit_model_path=implicit_model_path,
        explicit_packet_model_path=explicit_packet_model_path
    )
    print(f"Phase 1 completed. Best checkpoint: {phase1_checkpoint}")

    # Phase 2: Fine-tune explicit+packet model + fusion
    phase2_checkpoint = train_phase(
        config,
        phase=2,
        implicit_model_path=implicit_model_path,
        explicit_packet_model_path=explicit_packet_model_path,
        checkpoint_path=phase1_checkpoint
    )
    print(f"Phase 2 completed. Best checkpoint: {phase2_checkpoint}")

    # Phase 3: Fine-tune entire model (optional, use with caution)
    use_phase3 = input(
        "Do you want to run Phase 3 (full fine-tuning)? This may hurt performance if implicit model is already very good. (y/n): ")
    if use_phase3.lower() == 'y':
        phase3_checkpoint = train_phase(
            config,
            phase=3,
            implicit_model_path=implicit_model_path,
            explicit_packet_model_path=explicit_packet_model_path,
            checkpoint_path=phase2_checkpoint
        )
        print(f"Phase 3 completed. Best checkpoint: {phase3_checkpoint}")
        final_checkpoint = phase3_checkpoint
    else:
        final_checkpoint = phase2_checkpoint

    print(f"\nTraining completed! Final best model: {final_checkpoint}")


def train_explicit_packet_model(config):
    """Train the explicit+packet model separately"""
    from EPT_PPT_CA_model import ExplicitPacketModel

    print("Training Explicit+Packet Model...")

    # Create a separate trainer for explicit+packet model
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
            labels = batch['labels'][:, 0]
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
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['training']['num_epochs']
            )
            return [optimizer], [scheduler]

    # Setup data and train
    data_module = CombinedDataModule(config)
    data_module.setup()

    sample_batch = next(iter(data_module.train_dataloader()))

    explicit_model = ExplicitPacketTrainer(
        config,
        flow_dim=sample_batch['flow_features'].shape[-1],
        packet_dim=sample_batch['packets'].shape[-1],
        num_classes=config['data']['num_classes'],
        class_weights=data_module.class_weights,
        feature_info=data_module.feature_info
    )

    wandb_logger = WandbLogger(
        project=f"{config['logging']['experiment_name']}_explicit_packet",
        save_dir=config['logging']['save_dir']
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config['logging']['save_dir']}/explicit_packet",
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
    )

    trainer.fit(explicit_model, datamodule=data_module)
    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    main()