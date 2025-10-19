# hybrid_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from typing import Dict, Any, Tuple
import yaml


class HybridTransformerClassifier(pl.LightningModule):
    """
    Hybrid model that combines implicit and explicit transformer models.
    This version is corrected to statically define all layers in __init__
    to allow for correct checkpoint loading.
    """

    def __init__(self,
                 implicit_model_class,
                 explicit_model_class,
                 implicit_config: Dict[str, Any],  # FIX: We must pass configs in
                 explicit_config: Dict[str, Any],  # FIX: We must pass configs in
                 fusion_config: Dict[str, Any]):
        super().__init__()

        # FIX: Save all configs so load_from_checkpoint works
        # This allows PL to save these configs *into* the checkpoint file
        self.save_hyperparameters('implicit_config', 'explicit_config', 'fusion_config')

        # FIX: REMOVE hardcoded config loading from within __init__
        # The 'train.py' script will be responsible for loading these
        # and passing them in.

        # Initialize both models
        self.implicit_model = implicit_model_class(implicit_config)
        self.explicit_model = explicit_model_class(explicit_config)

        # Fusion configuration
        self.implicit_weight = fusion_config.get('implicit_weight', 0.7)
        self.explicit_weight = fusion_config.get('explicit_weight', 0.3)
        self.fusion_method = fusion_config.get('method', 'weighted_average')

        # --- FIX: START STATIC LAYER DEFINITION ---

        # Get representation dimensions from sub-model configs

        # Implicit Model Dimensions
        im_embed_dim = implicit_config['model']['embed_dim']
        # Check for protocol_embed_dim, default to embed_dim if not specified
        if 'protocol_embed_dim' not in implicit_config['model']:
            im_proto_dim = im_embed_dim
        else:
            im_proto_dim = implicit_config['model']['protocol_embed_dim']

        # The implicit model's representation is a concat of its embedding and protocol embedding
        im_repr_dim = im_embed_dim + im_proto_dim

        # Explicit Model Dimensions
        ex_repr_dim = explicit_config['model']['embed_dim']

        # Statically define the projection layer in __init__
        # This resolves the "Unexpected key(s) in state_dict: "implicit_proj.weight" error
        if im_repr_dim != ex_repr_dim:
            self.implicit_proj = nn.Linear(im_repr_dim, ex_repr_dim)
        else:
            self.implicit_proj = nn.Identity()  # No projection needed

        # The fused dimension will be the explicit model's dimension (after projection)
        fused_dim = ex_repr_dim

        # Get num_classes from one of the configs (they should be the same)
        num_classes = implicit_config['model']['num_classes']
        dropout = implicit_config['model']['dropout']

        # Statically define fusion layers
        if self.fusion_method == 'attention':
            self.fusion_attention = nn.MultiheadAttention(
                fused_dim, num_heads=4, dropout=dropout, batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(fused_dim)

        elif self.fusion_method == 'concat':
            # Concat dim is (projected implicit) + (explicit) = fused_dim + fused_dim
            self.fusion_proj = nn.Linear(fused_dim * 2, fused_dim)

        elif self.fusion_method == 'gated':
            self.gate_network = nn.Sequential(
                nn.Linear(fused_dim * 2, fused_dim),
                nn.ReLU(),
                nn.Linear(fused_dim, 2),
                nn.Softmax(dim=-1)
            )

        # Statically define the final classifier with the correct input dimension
        # This resolves the "size mismatch for final_classifier.0.weight" error

        # The error log showed:
        # Checkpoint (Trained): [64, 256] -> in=256, out=64
        # Current Model: [64, 64]        -> in=64, out=64
        # This tells us:
        # - The input to the classifier (fused_dim) should be 256
        # - The hidden dim of the classifier (output of first linear layer) should be 64

        # This 'classifier_hidden_dim' seems to be controlled by the *implicit* model's embed_dim
        classifier_hidden_dim = implicit_config['model']['embed_dim']

        self.final_classifier = nn.Sequential(
            nn.Linear(fused_dim, classifier_hidden_dim),  # In=256, Out=64 (based on error log)
            nn.LayerNorm(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes)  # In=64, Out=6
        )
        # --- FIX: END STATIC LAYER DEFINITION ---

        # Metrics
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes,
                                                            average='weighted')
        self.val_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes,
                                                          average='weighted')

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.lr = fusion_config.get('learning_rate', 1e-3)

    # ATTN: Modified to accept return_attn_weights
    def get_implicit_representation(self, features, protocol_ids, return_attn_weights: bool = False):
        """Get representation from implicit model before final classification"""
        if hasattr(self.implicit_model, 'get_representation'):
            # This is the ideal path, assuming implicit model was also updated
            rep, w = self.implicit_model.get_representation(features, protocol_ids,
                                                            return_attn_weights=return_attn_weights)
            return rep, w
        else:
            # Fallback for original implicit model
            with torch.no_grad():
                x = features.unsqueeze(-1)
                x = self.implicit_model.feature_embedding(x)
                x = self.implicit_model.pos_encoder(x)
                x = self.implicit_model.dropout(x)
                x = self.implicit_model.transformer(x)  # [0] if using CustomEncoder
                x = x.max(dim=1)[0]
                protocol_embed = self.implicit_model.protocol_embedding(protocol_ids)
                representation = torch.cat([x, protocol_embed], dim=1)
                return representation, None

    # ATTN: Modified to accept return_attn_weights
    def get_explicit_representation(self, features_dict, return_attn_weights: bool = False):
        """Get representation from explicit model before final classification"""
        if hasattr(self.explicit_model, 'get_representation'):
            # This is the ideal path
            representation, attn_weights_dict = self.explicit_model.get_representation(
                features_dict, return_attn_weights=return_attn_weights
            )
            return representation, attn_weights_dict
        else:
            # Fallback for original explicit model
            with torch.no_grad():
                # This is a HACK and depends on explicit model's forward pass
                # A proper get_representation is strongly recommended
                representation, _ = self.explicit_model(features_dict, return_attn_weights=False)
                return representation, None

    # ATTN: Modified forward to accept and return weights
    def forward(self, implicit_features, explicit_features_dict, protocol_ids, return_attn_weights: bool = False):
        """
        Forward pass through hybrid model
        """
        attn_weights_dict = {}

        # Get representations from both models
        implicit_repr, implicit_attns = self.get_implicit_representation(
            implicit_features, protocol_ids, return_attn_weights=return_attn_weights
        )
        explicit_repr, explicit_attns = self.get_explicit_representation(
            explicit_features_dict, return_attn_weights=return_attn_weights
        )

        if return_attn_weights:
            attn_weights_dict['implicit_model_attns'] = implicit_attns
            attn_weights_dict['explicit_model_attns'] = explicit_attns

        # FIX: Apply the pre-defined projection layer
        implicit_repr = self.implicit_proj(implicit_repr)

        # Fusion
        if self.fusion_method == 'weighted_average':
            fused_repr = (self.implicit_weight * implicit_repr +
                          self.explicit_weight * explicit_repr)

        elif self.fusion_method == 'attention':
            representations = torch.stack([implicit_repr, explicit_repr], dim=1)
            attended, fusion_attn = self.fusion_attention(
                representations, representations, representations, need_weights=return_attn_weights
            )
            fused_repr = self.fusion_norm(representations + attended).mean(dim=1)
            if return_attn_weights:
                attn_weights_dict['fusion_attention'] = fusion_attn

        elif self.fusion_method == 'concat':
            concatenated = torch.cat([implicit_repr, explicit_repr], dim=-1)
            fused_repr = self.fusion_proj(concatenated)

        elif self.fusion_method == 'gated':
            concatenated = torch.cat([implicit_repr, explicit_repr], dim=-1)
            gates = self.gate_network(concatenated)
            fused_repr = (gates[:, 0:1] * implicit_repr +
                          gates[:, 1:2] * explicit_repr)

        else:  # Default to weighted average
            fused_repr = (self.implicit_weight * implicit_repr +
                          self.explicit_weight * explicit_repr)

        # FIX: REMOVE dynamic classifier creation
        # if fused_repr.size(-1) != self.final_classifier[0].in_features: ...

        # Final classification
        logits = self.final_classifier(fused_repr)

        if return_attn_weights:
            return logits, attn_weights_dict
        else:
            return logits

    def training_step(self, batch, batch_idx):
        implicit_features, explicit_features_dict, protocol_ids, labels = batch

        logits = self(implicit_features, explicit_features_dict, protocol_ids)
        loss = self.criterion(logits, labels)

        # Update metrics (this doesn't log, just accumulates)
        self.train_acc(logits, labels)
        self.train_f1(logits, labels)

        # Log train_loss ON STEP for the progress bar
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log train_acc ON EPOCH and NOT on the progress bar
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=False)

        # Log train_f1 ON EPOCH and NOT on the progress bar
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        implicit_features, explicit_features_dict, protocol_ids, labels = batch

        logits = self(implicit_features, explicit_features_dict, protocol_ids)
        loss = self.criterion(logits, labels)

        # Update metrics (this doesn't log, just accumulates)
        self.val_acc(logits, labels)
        self.val_f1(logits, labels)

        # Log val_loss ON STEP for the progress bar
        # We set on_step=True here so it updates live during the validation dataloader run
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log val_acc ON EPOCH and NOT on the progress bar
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=False)

        # Log val_f1 ON EPOCH and NOT on the progress bar
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.fusion_config.get('learning_rate', 1e-3),
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        # Use PL's built-in way to get total steps
        total_steps = self.trainer.estimated_stepping_batches

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.fusion_config.get('learning_rate', 1e-3),
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos'
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def load_pretrained_weights(self, implicit_ckpt_path=None, explicit_ckpt_path=None):
        """Load pretrained weights from individual model checkpoints"""

        if implicit_ckpt_path:
            print(f"Loading implicit model weights from {implicit_ckpt_path}")
            implicit_ckpt = torch.load(implicit_ckpt_path, map_location=self.device)

            implicit_state = {}
            for key, value in implicit_ckpt['state_dict'].items():
                new_key = key.replace('implicit_model.', '') if 'implicit_model.' in key else key
                implicit_state[f'implicit_model.{new_key}'] = value

            self.load_state_dict(implicit_state, strict=False)
            print("✅ Implicit model weights loaded successfully")

        if explicit_ckpt_path:
            print(f"Loading explicit model weights from {explicit_ckpt_path}")
            explicit_ckpt = torch.load(explicit_ckpt_path, map_location=self.device)

            explicit_state = {}
            for key, value in explicit_ckpt['state_dict'].items():
                new_key = key.replace('explicit_model.', '') if 'explicit_model.' in key else key
                explicit_state[f'explicit_model.{new_key}'] = value

            self.load_state_dict(explicit_state, strict=False)
            print("✅ Explicit model weights loaded successfully")