import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

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

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head (after mean pooling)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        # Input: (batch, num_features)
        x = x.unsqueeze(-1)  # -> (batch, num_features, 1)
        x = self.feature_embedding(x)  # -> (batch, num_features, embed_dim)
        x = self.pos_encoder(x)        # -> (batch, num_features, embed_dim)
        x = self.dropout(x)
        x = self.transformer(x)        # -> (batch, num_features, embed_dim)

        # Mean pooling across sequence (dim=1)
        x = x.mean(dim=1)  # -> (batch, embed_dim)

        logits = self.classifier(x)  # -> (batch, num_classes)
        return logits

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_acc(logits, labels)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_acc(logits, labels)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.lr)

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
