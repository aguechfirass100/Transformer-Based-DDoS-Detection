import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import math


class SimplifiedTemporalAttention(nn.Module):
    """Simplified temporal attention with better sequence modeling"""

    def __init__(self, input_dim, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Simple input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Simple temporal pattern recognition
        self.temporal_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim // 4)

        # Simple FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.shape[0]

        # Project input
        x_proj = self.input_proj(x)

        # Create temporal sequence - simple approach
        seq_len = 8  # Fixed sequence length
        x_seq = x_proj.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, embed_dim)

        # Self-attention
        attn_out, _ = self.self_attn(x_seq, x_seq, x_seq)

        # Simple temporal convolution
        conv_input = attn_out.transpose(1, 2)  # (batch, embed_dim, seq_len)
        conv_out = self.temporal_conv(conv_input)
        conv_out = conv_out.transpose(1, 2)  # (batch, seq_len, embed_dim)

        # Combine and process
        combined = self.norm(attn_out + conv_out)
        ffn_out = self.ffn(combined)
        output = combined + ffn_out

        # Global average pooling
        return self.dropout(output.mean(dim=1))


class SimplifiedSpatialAttention(nn.Module):
    """Simplified spatial attention"""

    def __init__(self, input_dim, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Simple projections
        mid_point = input_dim // 2
        self.source_proj = nn.Linear(mid_point, embed_dim // 2)
        self.dest_proj = nn.Linear(mid_point, embed_dim // 2)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim // 2, num_heads // 2, dropout=dropout, batch_first=True
        )

        # Final processing
        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        # Split features
        mid_point = self.input_dim // 2
        source_feats = x[:, :mid_point]
        dest_feats = x[:, mid_point:]

        # Project
        source_emb = self.source_proj(source_feats).unsqueeze(1)
        dest_emb = self.dest_proj(dest_feats).unsqueeze(1)

        # Cross-attention
        source_attn, _ = self.cross_attn(source_emb, dest_emb, dest_emb)
        dest_attn, _ = self.cross_attn(dest_emb, source_emb, source_emb)

        # Combine
        combined = torch.cat([source_attn, dest_attn], dim=-1).squeeze(1)

        # Final processing
        output = self.final_proj(combined)
        return self.norm(output)


class SimplifiedProtocolAttention(nn.Module):
    """Simplified protocol attention"""

    def __init__(self, input_dim, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Protocol embedding
        self.protocol_embedding = nn.Embedding(256, embed_dim // 4)

        # Feature projection
        self.feature_proj = nn.Linear(input_dim - 1, embed_dim * 3 // 4)

        # Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Processing
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        # Extract protocol and features
        protocol_id = x[:, 0].long().clamp(0, 255)
        other_features = x[:, 1:] if self.input_dim > 1 else torch.zeros(batch_size, 1, device=x.device)

        # Get embeddings
        protocol_emb = self.protocol_embedding(protocol_id)
        if other_features.size(1) > 0:
            features_emb = self.feature_proj(other_features)
        else:
            features_emb = torch.zeros(batch_size, self.embed_dim * 3 // 4, device=x.device)

        # Combine
        combined = torch.cat([protocol_emb, features_emb], dim=-1).unsqueeze(1)

        # Attention
        attn_out, _ = self.self_attn(combined, combined, combined)
        attn_out = attn_out.squeeze(1)

        # FFN
        ffn_out = self.ffn(attn_out)
        output = self.norm(attn_out + ffn_out)

        return output


class SimplifiedStatisticalAttention(nn.Module):
    """Simplified statistical attention"""

    def __init__(self, input_dim, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Feature importance learning
        self.feature_weights = nn.Parameter(torch.ones(input_dim))

        # Global processing
        self.global_proj = nn.Linear(input_dim, embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Simple feature grouping (if enough features)
        if input_dim >= 8:
            group_size = input_dim // 4
            self.group1_proj = nn.Linear(group_size, embed_dim // 4)
            self.group2_proj = nn.Linear(group_size, embed_dim // 4)
            self.group3_proj = nn.Linear(group_size, embed_dim // 4)
            self.group4_proj = nn.Linear(input_dim - 3 * group_size, embed_dim // 4)
            self.has_groups = True
            self.group_size = group_size
        else:
            self.has_groups = False

        # Final processing
        final_input_dim = embed_dim + (embed_dim if self.has_groups else 0)
        self.final_proj = nn.Sequential(
            nn.Linear(final_input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        # Apply feature importance
        weighted_x = x * F.softmax(self.feature_weights, dim=0)

        # Global processing
        global_emb = self.global_proj(weighted_x).unsqueeze(1)
        global_attn, _ = self.self_attn(global_emb, global_emb, global_emb)
        global_out = global_attn.squeeze(1)

        # Group processing if available
        if self.has_groups:
            g1 = self.group1_proj(weighted_x[:, :self.group_size])
            g2 = self.group2_proj(weighted_x[:, self.group_size:2 * self.group_size])
            g3 = self.group3_proj(weighted_x[:, 2 * self.group_size:3 * self.group_size])
            g4 = self.group4_proj(weighted_x[:, 3 * self.group_size:])
            group_out = torch.cat([g1, g2, g3, g4], dim=-1)
            combined = torch.cat([global_out, group_out], dim=-1)
        else:
            combined = global_out

        # Final processing
        output = self.final_proj(combined)
        return self.norm(output)


class SimplifiedMultiAttentionTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # Simplified configuration
        embed_dim = config['model']['embed_dim']
        num_heads = config['model']['num_heads']
        num_layers = config['model']['num_layers']
        ffn_dim = config['model']['ffn_dim']
        dropout = config['model']['dropout']
        num_classes = config['model']['num_classes']
        lr = float(config['training']['learning_rate'])

        # Feature dimensions
        temp_dim = len(config['feature_groups']['temporal'])
        spatial_dim = len(config['feature_groups']['spatial'])
        proto_dim = len(config['feature_groups']['protocol'])
        stat_dim = len(config['feature_groups']['statistical'])

        # Initialize simplified attention modules
        self.temporal_attn = SimplifiedTemporalAttention(temp_dim, embed_dim, num_heads // 2, dropout)
        self.spatial_attn = SimplifiedSpatialAttention(spatial_dim, embed_dim, num_heads // 2, dropout)
        self.protocol_attn = SimplifiedProtocolAttention(proto_dim, embed_dim, num_heads // 2, dropout)
        self.statistical_attn = SimplifiedStatisticalAttention(stat_dim, embed_dim, num_heads // 2, dropout)

        # Simple cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        # Simplified transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(embed_dim, num_classes)
        )

        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes,
                                                            average='weighted')
        self.val_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes,
                                                          average='weighted')

        self.lr = lr

    def forward(self, features, protocol_ids=None):
        # Apply attention modules
        temp_out = self.temporal_attn(features['temporal'])
        spatial_out = self.spatial_attn(features['spatial'])
        proto_out = self.protocol_attn(features['protocol'])
        stat_out = self.statistical_attn(features['statistical'])

        # Create sequence for cross-attention
        feature_sequence = torch.stack([temp_out, spatial_out, proto_out, stat_out], dim=1)

        # Cross-attention
        cross_out, _ = self.cross_attention(feature_sequence, feature_sequence, feature_sequence)
        cross_out = self.cross_norm(feature_sequence + cross_out)

        # Transformer processing
        transformer_out = self.transformer(cross_out)

        # Global average pooling
        pooled = transformer_out.mean(dim=1)

        # Classification
        logits = self.classifier(pooled)
        return logits

    def training_step(self, batch, batch_idx):
        features, protocol_ids, labels = batch
        logits = self(features, protocol_ids)
        loss = self.criterion(logits, labels)

        self.train_acc(logits, labels)
        self.train_f1(logits, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        features, protocol_ids, labels = batch
        logits = self(features, protocol_ids)
        loss = self.criterion(logits, labels)

        self.val_acc(logits, labels)
        self.val_f1(logits, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches if hasattr(self, 'trainer') else 1000,
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