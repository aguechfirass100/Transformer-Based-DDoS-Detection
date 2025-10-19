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

        # Handle empty input
        if input_dim < 1:
            input_dim = 1  # Minimum dimension

        # Simple input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # Multi-head attention - ensure at least 1 head
        num_heads = max(1, num_heads)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Simple temporal pattern recognition - ensure valid groups
        groups = max(1, min(embed_dim // 4, embed_dim))
        self.temporal_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=groups)

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

        # Handle empty input
        if x.size(1) == 0:
            x = torch.zeros(batch_size, 1, device=x.device)

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

        # Handle case where input_dim is very small
        if input_dim < 2:
            # If less than 2 features, just do simple projection
            self.simple_mode = True
            self.simple_proj = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.simple_mode = False
            # Simple projections - ensure at least 1 feature per part
            mid_point = max(1, input_dim // 2)
            remaining = input_dim - mid_point

            self.source_proj = nn.Linear(mid_point, embed_dim // 2)
            self.dest_proj = nn.Linear(remaining, embed_dim // 2)

            # Cross-attention
            self.cross_attn = nn.MultiheadAttention(
                embed_dim // 2, max(1, num_heads // 2), dropout=dropout, batch_first=True
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

        if self.simple_mode:
            # Simple projection for very small input
            output = self.simple_proj(x)
            return self.norm(output)

        # Split features
        mid_point = max(1, self.input_dim // 2)
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

        if input_dim < 1:
            # Handle empty input
            self.simple_mode = True
            self.simple_proj = nn.Linear(1, embed_dim)  # Dummy input
        elif input_dim == 1:
            # Only protocol, no other features
            self.protocol_embedding = nn.Embedding(256, embed_dim)
            self.feature_proj = None
            self.simple_mode = False
        else:
            # Protocol + other features
            self.protocol_embedding = nn.Embedding(256, embed_dim // 4)
            self.feature_proj = nn.Linear(input_dim - 1, embed_dim * 3 // 4)
            self.simple_mode = False

        # Attention (only if not simple mode)
        if not self.simple_mode:
            self.self_attn = nn.MultiheadAttention(
                embed_dim, max(1, num_heads), dropout=dropout, batch_first=True
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

        if self.simple_mode:
            # Handle empty or problematic input
            dummy_input = torch.ones(batch_size, 1, device=x.device)
            output = self.simple_proj(dummy_input)
            return self.norm(output)

        if self.input_dim == 1:
            # Only protocol
            protocol_id = x[:, 0].long().clamp(0, 255)
            output = self.protocol_embedding(protocol_id)
            return self.norm(output)

        # Extract protocol and features
        protocol_id = x[:, 0].long().clamp(0, 255)
        other_features = x[:, 1:]

        # Get embeddings
        protocol_emb = self.protocol_embedding(protocol_id)
        features_emb = self.feature_proj(other_features)

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

        # Get dimensions from config
        flow_dim = config['data']['flow_features_dim']
        num_classes = config['data']['num_classes']

        # Model config
        model_config = config['model']
        embed_dim = model_config['flow_context_dim']
        num_heads = model_config['explicit_heads']['num_heads']
        dropout = model_config['packet_implicit']['dropout']

        self.output_dim = embed_dim

        # Fixed slicing based on the actual flow features dimensions
        # Adjust these based on your actual feature dimensions
        self.temp_slice = slice(0, min(8, flow_dim))  # temporal features
        temp_size = min(8, flow_dim)

        self.spatial_slice = slice(temp_size, min(temp_size + 3, flow_dim))  # spatial features
        spatial_size = min(3, max(0, flow_dim - temp_size))

        self.protocol_slice = slice(temp_size + spatial_size,
                                    min(temp_size + spatial_size + 2, flow_dim))  # protocol features
        protocol_size = min(2, max(0, flow_dim - temp_size - spatial_size))

        # Remaining features for statistical attention
        stat_start = temp_size + spatial_size + protocol_size
        self.stat_slice = slice(stat_start, flow_dim)
        stat_size = max(1, flow_dim - stat_start)  # At least 1 feature

        print(f"Feature slicing: temp={temp_size}, spatial={spatial_size}, protocol={protocol_size}, stat={stat_size}")

        # Initialize simplified attention modules with correct dimensions
        self.temporal_attn = SimplifiedTemporalAttention(temp_size, embed_dim, num_heads // 2, dropout)
        self.spatial_attn = SimplifiedSpatialAttention(spatial_size, embed_dim, num_heads // 2, dropout)
        self.protocol_attn = SimplifiedProtocolAttention(protocol_size, embed_dim, num_heads // 2, dropout)
        self.statistical_attn = SimplifiedStatisticalAttention(stat_size, embed_dim, num_heads // 2, dropout)

        # Simple cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        # Simplified transformer
        cross_config = model_config['cross_attention']
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=cross_config['num_heads'],
            dim_feedforward=cross_config['hidden_dim'],
            dropout=cross_config['dropout'],
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cross_config['num_layers'])

        # Simplified classifier
        classifier_config = model_config['classifier']
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, classifier_config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(classifier_config['dropout']),
            nn.Linear(classifier_config['hidden_dim'], num_classes)
        )

        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes,
                                                            average='weighted')
        self.val_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes,
                                                          average='weighted')

        self.lr = config['training']['learning_rate']

    def forward(self, input_data):
        """Forward pass - handles both batch dict and tensor input"""
        if isinstance(input_data, dict):
            # Called with batch dictionary from dataloader
            flow_features = input_data['flow_features'].float()
            flow_mask = input_data['mask'].float()  # [batch_size, flows_per_batch]
        else:
            # Called with tensor directly from parent model
            flow_features = input_data.float()  # [batch_size, flows_per_batch, flow_features_dim]
            # Create default mask (all flows valid)
            B, F, D = flow_features.shape
            flow_mask = torch.ones(B, F, device=flow_features.device, dtype=torch.float32)

        B, F, D = flow_features.shape

        # Flatten flows for processing: [batch_size * flows_per_batch, flow_features_dim]
        flow_flat = flow_features.reshape(B * F, D)

        # Slice features for different attention heads
        temporal_feats = flow_flat[:, self.temp_slice]
        spatial_feats = flow_flat[:, self.spatial_slice]
        protocol_feats = flow_flat[:, self.protocol_slice]
        stat_feats = flow_flat[:, self.stat_slice]

        # Apply attention modules
        temp_out = self.temporal_attn(temporal_feats)  # [B*F, embed_dim]
        spatial_out = self.spatial_attn(spatial_feats)  # [B*F, embed_dim]
        proto_out = self.protocol_attn(protocol_feats)  # [B*F, embed_dim]
        stat_out = self.statistical_attn(stat_feats)  # [B*F, embed_dim]

        # Stack into sequence: [B*F, 4, embed_dim]
        feature_sequence = torch.stack([temp_out, spatial_out, proto_out, stat_out], dim=1)

        # Cross-attention and transformer operate on [batch_flat, seq_len, embed_dim]
        cross_out, _ = self.cross_attention(feature_sequence, feature_sequence, feature_sequence)
        cross_out = self.cross_norm(feature_sequence + cross_out)

        transformer_out = self.transformer(cross_out)

        # Global average pooling over the 4 attention heads -> [B*F, embed_dim]
        pooled = transformer_out.mean(dim=1)

        # Reshape back to [B, F, embed_dim]
        pooled = pooled.view(B, F, -1)

        # Apply flow mask and pool over flows
        flow_mask_expanded = flow_mask.unsqueeze(-1)  # [B, F, 1]
        masked_output = pooled * flow_mask_expanded

        # Don't pool over flows yet - return 3D tensor to match packet transformer
        # final_pooled = masked_output.sum(dim=1) / (flow_mask_expanded.sum(dim=1) + 1e-8)  # [B, embed_dim]

        # Return the 3D tensor: [batch_size, flows_per_batch, embed_dim]
        return masked_output

    def training_step(self, batch, batch_idx):
        # This model now returns representations, not logits
        # So we don't implement training here - it's handled by the parent model
        raise NotImplementedError("This model is used as a component, training handled by parent model")

    def validation_step(self, batch, batch_idx):
        # This model now returns representations, not logits
        # So we don't implement validation here - it's handled by the parent model
        raise NotImplementedError("This model is used as a component, validation handled by parent model")

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