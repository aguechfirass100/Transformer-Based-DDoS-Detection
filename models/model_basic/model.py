import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from typing import Tuple, Dict, Optional, List  # ATTN: Added imports


# ATTN: New module to replace nn.TransformerEncoder
class CustomTransformerEncoder(nn.Module):
    """
    A custom TransformerEncoder that allows returning attention weights
    from each layer. This replicates the one from the explicit model.
    """

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # Create a list of encoder layers
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, return_attn_weights: bool = False) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        output = src
        all_attn_weights = []

        for layer in self.layers:
            # Replicating the forward pass of nn.TransformerEncoderLayer
            # to gain access to self_attn's weights.

            # 1. Self-Attention
            # We pass need_weights=return_attn_weights to the self_attn module
            sa_out, attn_weights = layer.self_attn(
                output, output, output, need_weights=return_attn_weights
            )
            output = output + layer.dropout1(sa_out)
            output = layer.norm1(output)

            # 2. Feed-Forward
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(output))))
            output = output + layer.dropout2(ff_out)
            output = layer.norm2(output)

            if return_attn_weights:
                all_attn_weights.append(attn_weights)

        if return_attn_weights:
            return output, all_attn_weights
        else:
            return output, None


class TransformerClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        embed_dim = config['model']['embed_dim']
        num_heads = config['model']['num_heads']
        num_layers = config['model']['num_layers']
        ffn_dim = config['model']['ffn_dim']
        dropout = config['model']['dropout']
        num_classes = config['model']['num_classes']
        lr = float(config['training']['learning_rate'])
        self.max_epochs = int(config['training']['epochs'])

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embed_dim)

        # Embed scalar features
        self.feature_embedding = nn.Linear(1, embed_dim)

        # Protocol Embedding
        self.num_protocols = 18  # max(protocol_id) + 1 — you can tune this
        self.protocol_embed_dim = embed_dim  # or use smaller dim if you like
        self.protocol_embedding = nn.Embedding(self.num_protocols, int(self.protocol_embed_dim))

        # ATTN: Replaced nn.TransformerEncoder with CustomTransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation='relu'  # ATTN: Explicitly added activation
        )
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer = CustomTransformerEncoder(encoder_layer, num_layers=num_layers)
        # ATTN: End of replacement

        # Classification head (after mean pooling)
        self.classifier = nn.Sequential(
            # nn.Linear(embed_dim, embed_dim),
            nn.Linear(embed_dim + self.protocol_embed_dim, embed_dim),  # embed_dim + protocol_embed_dim
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    # ATTN: Modified forward
    def forward(self, x, protocol_ids, return_attn_weights: bool = False):
        assert protocol_ids.max() < self.num_protocols, "Protocol ID exceeds num_protocols"
        assert protocol_ids.min() >= 0, "Protocol ID less than 0"

        # Input: (batch, num_features)
        x = x.unsqueeze(-1)  # (batch, num_features, 1)
        x = self.feature_embedding(x)  # (batch, num_features, embed_dim)

        x = self.pos_encoder(x)  # (batch, num_features, embed_dim)
        x = self.dropout(x)

        # ATTN: Capture weights from transformer
        x, attn_weights = self.transformer(x, return_attn_weights=return_attn_weights)

        # Max pooling across sequence (dim=1)
        x = x.max(dim=1)[0]

        # Get protocol embedding and concatenate
        protocol_embed = self.protocol_embedding(protocol_ids)  # (batch, protocol_embed_dim)
        x = torch.cat([x, protocol_embed], dim=1)  # (batch, embed_dim + protocol_embed_dim)

        logits = self.classifier(x)  # (batch, num_classes)

        # ATTN: Return weights if requested
        if return_attn_weights:
            attn_dict = {'transformer_encoder_layers': attn_weights}  # attn_weights is a list
            return logits, attn_dict
        else:
            return logits

    # ATTN: Modified get_representation
    def get_representation(self, x, protocol_ids, return_attn_weights: bool = False):
        """Get representation before final classification"""
        assert protocol_ids.max() < self.num_protocols, "Protocol ID exceeds num_protocols"
        assert protocol_ids.min() >= 0, "Protocol ID less than 0"

        # Input: (batch, num_features)
        x = x.unsqueeze(-1)  # (batch, num_features, 1)
        x = self.feature_embedding(x)  # (batch, num_features, embed_dim)

        x = self.pos_encoder(x)  # (batch, num_features, embed_dim)
        x = self.dropout(x)

        # ATTN: Capture weights from transformer
        x, attn_weights = self.transformer(x, return_attn_weights=return_attn_weights)

        # Max pooling across sequence (dim=1)
        x = x.max(dim=1)[0]

        # Get protocol embedding and concatenate
        protocol_embed = self.protocol_embedding(protocol_ids)  # (batch, protocol_embed_dim)
        representation = torch.cat([x, protocol_embed], dim=1)  # (batch, embed_dim + protocol_embed_dim)

        # ATTN: Return weights if requested
        if return_attn_weights:
            attn_dict = {'transformer_encoder_layers': attn_weights}  # attn_weights is a list
            return representation, attn_dict
        else:
            return representation, None

    def training_step(self, batch, batch_idx):
        features, protocol_ids, labels = batch
        # ATTN: Call forward normally
        logits = self(features, protocol_ids)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_acc(logits, labels)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, protocol_ids, labels = batch
        # ATTN: Call forward normally
        logits = self(features, protocol_ids)
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_acc(logits, labels)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }


class PositionalEncoding(nn.Module):
    # ... (This class is unchanged) ...
    def __init__(self, embed_dim, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)  # (max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x