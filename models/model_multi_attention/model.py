import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import math
from typing import Tuple, Dict, Optional, List  # ATTN: Added imports


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

    # ATTN: Modified forward
    def forward(self, x, return_attn_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = x.shape[0]

        # Project input
        x_proj = self.input_proj(x)

        # Create temporal sequence - simple approach
        seq_len = 8  # Fixed sequence length
        x_seq = x_proj.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, embed_dim)

        # Self-attention
        # ATTN: Capture weights
        attn_out, attn_weights = self.self_attn(
            x_seq, x_seq, x_seq, need_weights=return_attn_weights
        )

        # Simple temporal convolution
        conv_input = attn_out.transpose(1, 2)  # (batch, embed_dim, seq_len)
        conv_out = self.temporal_conv(conv_input)
        conv_out = conv_out.transpose(1, 2)  # (batch, seq_len, embed_dim)

        # Combine and process
        combined = self.norm(attn_out + conv_out)
        ffn_out = self.ffn(combined)
        output = combined + ffn_out

        # Global average pooling
        pooled_output = self.dropout(output.mean(dim=1))

        # ATTN: Return weights
        if return_attn_weights:
            return pooled_output, attn_weights
        else:
            return pooled_output, None


class SimplifiedSpatialAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Dynamically split input based on number of features
        if input_dim == 3:
            # 1 source, 2 destination
            self.source_proj = nn.Linear(1, embed_dim // 2)
            self.dest_proj = nn.Linear(2, embed_dim // 2)
        else:
            # fallback: split in half
            mid_point = input_dim // 2
            self.source_proj = nn.Linear(mid_point, embed_dim // 2)
            self.dest_proj = nn.Linear(input_dim - mid_point, embed_dim // 2)

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

    # ATTN: Modified forward
    def forward(self, x, return_attn_weights: bool = False) -> Tuple[
        torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size = x.shape[0]

        if self.input_dim == 3:
            source_feats = x[:, :1]  # first feature
            dest_feats = x[:, 1:]  # remaining 2 features
        else:
            mid_point = self.input_dim // 2
            source_feats = x[:, :mid_point]
            dest_feats = x[:, mid_point:]

        # Project
        source_emb = self.source_proj(source_feats).unsqueeze(1)
        dest_emb = self.dest_proj(dest_feats).unsqueeze(1)

        # Cross-attention
        # ATTN: Capture weights
        source_attn, source_attn_weights = self.cross_attn(
            source_emb, dest_emb, dest_emb, need_weights=return_attn_weights
        )
        dest_attn, dest_attn_weights = self.cross_attn(
            dest_emb, source_emb, source_emb, need_weights=return_attn_weights
        )

        # Combine
        combined = torch.cat([source_attn, dest_attn], dim=-1).squeeze(1)

        # Final processing
        output = self.final_proj(combined)

        # ATTN: Return weights
        if return_attn_weights:
            return self.norm(output), (source_attn_weights, dest_attn_weights)
        else:
            return self.norm(output), None


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

    # ATTN: Modified forward
    def forward(self, x, return_attn_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        # ATTN: Capture weights
        attn_out, attn_weights = self.self_attn(
            combined, combined, combined, need_weights=return_attn_weights
        )
        attn_out = attn_out.squeeze(1)

        # FFN
        ffn_out = self.ffn(attn_out)
        output = self.norm(attn_out + ffn_out)

        # ATTN: Return weights
        if return_attn_weights:
            return output, attn_weights
        else:
            return output, None


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

    # ATTN: Modified forward
    def forward(self, x, return_attn_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = x.shape[0]

        # Apply feature importance
        weighted_x = x * F.softmax(self.feature_weights, dim=0)

        # Global processing
        global_emb = self.global_proj(weighted_x).unsqueeze(1)
        # ATTN: Capture weights
        global_attn, attn_weights = self.self_attn(
            global_emb, global_emb, global_emb, need_weights=return_attn_weights
        )
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

        # ATTN: Return weights
        if return_attn_weights:
            return self.norm(output), attn_weights
        else:
            return self.norm(output), None


# ATTN: New module to replace nn.TransformerEncoder
class CustomTransformerEncoder(nn.Module):
    """
    A custom TransformerEncoder that allows returning attention weights
    from each layer.
    """

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # PyTorch 2.0+ uses nn.ModuleList, older versions might use nn.ModuleList
        # self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        # A simpler way that works:
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, return_attn_weights: bool = False) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        output = src
        all_attn_weights = []

        for layer in self.layers:
            # We must modify the nn.TransformerEncoderLayer to accept this flag
            # or, more simply, we assume it's a standard one and we access
            # its 'self_attn' module directly if we need weights.

            # This is tricky. The `nn.TransformerEncoderLayer` forward pass
            # does not take `need_weights`.
            # We must replicate the forward pass of TransformerEncoderLayer
            # to gain access to self_attn's weights.

            # --- Replicating TransformerEncoderLayer forward ---
            # 1. Self-Attention
            sa_out, attn_weights = layer.self_attn(
                output, output, output, need_weights=return_attn_weights
            )
            output = output + layer.dropout1(sa_out)
            output = layer.norm1(output)

            # 2. Feed-Forward
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(output))))
            output = output + layer.dropout2(ff_out)
            output = layer.norm2(output)
            # --- End of replication ---

            if return_attn_weights:
                all_attn_weights.append(attn_weights)

        if return_attn_weights:
            return output, all_attn_weights
        else:
            return output, None


class SimplifiedMultiAttentionTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # Configuration
        embed_dim = config['model']['embed_dim']
        num_heads = config['model']['num_heads']
        num_layers = config['model']['num_layers']
        ffn_dim = config['model']['ffn_dim']
        dropout = config['model']['dropout']
        num_classes = config['model']['num_classes']
        lr = float(config['training']['learning_rate'])

        self.output_dim = embed_dim
        self.feature_groups = config.get('feature_groups', {
            'temporal': [],
            'spatial': [],
            'protocol': [],
            'statistical': []
        })

        temp_dim = len(self.feature_groups['temporal'])
        spatial_dim = len(self.feature_groups['spatial'])
        proto_dim = len(self.feature_groups['protocol'])
        stat_dim = len(self.feature_groups['statistical'])

        # Initialize attention modules (keeping your existing ones)
        self.temporal_attn = SimplifiedTemporalAttention(temp_dim, embed_dim, num_heads // 2, dropout)
        self.spatial_attn = SimplifiedSpatialAttention(spatial_dim, embed_dim, num_heads // 2, dropout)
        self.protocol_attn = SimplifiedProtocolAttention(proto_dim, embed_dim, num_heads // 2, dropout)
        self.statistical_attn = SimplifiedStatisticalAttention(stat_dim, embed_dim, num_heads // 2, dropout)

        # FEATURE IMPORTANCE-BASED HEAD WEIGHTS
        self.init_head_weights_from_feature_importance()
        self.head_weights = nn.Parameter(self.base_head_weights.clone())
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Cross-attention with weighted combination
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        # ATTN: Replaced nn.TransformerEncoder with CustomTransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer = CustomTransformerEncoder(encoder_layer, num_layers=num_layers)
        # ATTN: End of replacement

        # Classifier
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

    def init_head_weights_from_feature_importance(self):
        """Initialize head weights based on integrated gradients feature importance"""
        # ... (no changes in this function) ...
        feature_importance = {
            'Timestamp': 0.23276792,
            'Min Packet Length': 0.014600158,
            'Average Packet Size': 0.007562412,
            'Fwd Packet Length Min': 0.005694733,
            'Fwd Packet Length Max': 0.0031518633,
            'Fwd IAT Mean': 0.0031303843,
            'Total Length of Fwd Packets': 0.002995217,
            'Fwd Packets/s': 0.002587274,
            'Flow Duration': 0.0022294908,
            'Packet Length Std': 0.0022104152,
            'Destination Port': 0.002160932,
            'Max Packet Length': 0.0020838114,
            'Fwd IAT Max': 0.0017841208,
            'Flow IAT Max': 0.0017546371,
            'Fwd IAT Total': 0.0014801916,
            'min_seg_size_forward': 0.0012240693,
            'Flow IAT Mean': 0.0010086567,
            'Fwd Packet Length Std': 0.0008855116,
            'Subflow Fwd Bytes': 0.0008646247,
            'Fwd Header Length': 0.00065524224,
            'ACK Flag Count': 0.0005464996,
            'Init_Win_bytes_forward': 0.00044921885,
            'Source IP': 0.00022200683,
            'Destination IP': 0.000118976604,
            'Flow IAT Min': 0.00010457172,
        }
        temporal_features = self.feature_groups['temporal']
        spatial_features = self.feature_groups['spatial']
        protocol_features = self.feature_groups['protocol']
        statistical_features = self.feature_groups['statistical']

        def calc_group_importance(features):
            if not features:
                return 0.001
            total = sum(feature_importance.get(feat, 0.001) for feat in features)
            return total / len(features)

        temporal_weight = calc_group_importance(temporal_features)
        spatial_weight = calc_group_importance(spatial_features)
        protocol_weight = calc_group_importance(protocol_features)
        statistical_weight = calc_group_importance(statistical_features)
        weights = torch.tensor([temporal_weight, spatial_weight, protocol_weight, statistical_weight])
        self.base_head_weights = weights / weights.sum()
        print(f"Initialized head weights based on feature importance:")
        print(f"  Temporal: {self.base_head_weights[0]:.4f}")
        print(f"  Spatial: {self.base_head_weights[1]:.4f}")
        print(f"  Protocol: {self.base_head_weights[2]:.4f}")
        print(f"  Statistical: {self.base_head_weights[3]:.4f}")

    # ... (no changes in this function) ...

    def get_dynamic_head_weights(self):
        """Get current head weights with temperature scaling"""
        return F.softmax(self.head_weights / self.temperature, dim=0)

    # ATTN: Created new function to separate representation from classification
    def get_representation(self, features, return_attn_weights: bool = False):
        """
        Runs the model forward pass up to the representation layer,
        optionally returning all attention weights.
        """
        # ATTN: Dictionary to store all weights
        attn_weights_dict = {}

        # ATTN: Call sub-modules and capture weights
        temp_out, temp_attn = self.temporal_attn(
            features['temporal'], return_attn_weights=return_attn_weights
        )
        spatial_out, spatial_attn = self.spatial_attn(
            features['spatial'], return_attn_weights=return_attn_weights
        )
        proto_out, proto_attn = self.protocol_attn(
            features['protocol'], return_attn_weights=return_attn_weights
        )
        stat_out, stat_attn = self.statistical_attn(
            features['statistical'], return_attn_weights=return_attn_weights
        )

        if return_attn_weights:
            attn_weights_dict['temporal_self_attn'] = temp_attn
            attn_weights_dict['spatial_cross_attn'] = spatial_attn  # This is a tuple
            attn_weights_dict['protocol_self_attn'] = proto_attn
            attn_weights_dict['statistical_self_attn'] = stat_attn

        # Get current head weights
        current_weights = self.get_dynamic_head_weights()

        # Apply weights to head outputs before stacking
        weighted_temp = temp_out * current_weights[0]
        weighted_spatial = spatial_out * current_weights[1]
        weighted_proto = proto_out * current_weights[2]
        weighted_stat = stat_out * current_weights[3]

        # Create weighted sequence
        feature_sequence = torch.stack([weighted_temp, weighted_spatial, weighted_proto, weighted_stat], dim=1)

        # Cross-attention
        # ATTN: Capture cross-attention weights
        cross_out, cross_attn_weights = self.cross_attention(
            feature_sequence, feature_sequence, feature_sequence, need_weights=return_attn_weights
        )
        cross_out = self.cross_norm(feature_sequence + cross_out)

        if return_attn_weights:
            attn_weights_dict['feature_group_cross_attn'] = cross_attn_weights

        # Transformer processing
        # ATTN: Capture internal transformer layer weights
        transformer_out, transformer_layer_attns = self.transformer(
            cross_out, return_attn_weights=return_attn_weights
        )

        if return_attn_weights:
            attn_weights_dict['transformer_encoder_layers'] = transformer_layer_attns  # This is a list

        # Weighted pooling instead of simple average
        pooled = (transformer_out * current_weights.view(1, -1, 1)).sum(dim=1)

        if return_attn_weights:
            return pooled, attn_weights_dict
        else:
            return pooled, None

    # ATTN: Modified forward to use get_representation
    def forward(self, features, protocol_ids=None, return_attn_weights: bool = False):

        representation, attn_weights_dict = self.get_representation(
            features, return_attn_weights=return_attn_weights
        )

        # Classification
        logits = self.classifier(representation)

        if return_attn_weights:
            return logits, attn_weights_dict
        else:
            return logits

    def training_step(self, batch, batch_idx):
        features, protocol_ids, labels = batch
        # ATTN: Pass return_attn_weights=False (default)
        logits = self(features, protocol_ids)
        loss = self.criterion(logits, labels)

        # ... (rest of the function is unchanged) ...
        weight_reg = 0.01 * torch.var(self.head_weights)
        loss = loss + weight_reg
        self.train_acc(logits, labels)
        self.train_f1(logits, labels)
        current_weights = self.get_dynamic_head_weights()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        self.log("weight_temporal", current_weights[0], on_step=False, on_epoch=True)
        self.log("weight_spatial", current_weights[1], on_step=False, on_epoch=True)
        self.log("weight_protocol", current_weights[2], on_step=False, on_epoch=True)
        self.log("weight_statistical", current_weights[3], on_step=False, on_epoch=True)
        self.log("temperature", self.temperature, on_step=False, on_epoch=True)
        return loss
        # ... (rest of the function is unchanged) ...

    def validation_step(self, batch, batch_idx):
        features, protocol_ids, labels = batch
        # ATTN: Pass return_attn_weights=False (default)
        logits = self(features, protocol_ids)
        loss = self.criterion(logits, labels)
        # ... (rest of the function is unchanged) ...
        self.val_acc(logits, labels)
        self.val_f1(logits, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)
        # ... (rest of the function is unchanged) ...

    def configure_optimizers(self):
        # ... (no changes in this function) ...
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
        # ... (no changes in this function) ...

    def get_head_weights_summary(self):
        # ... (no changes in this function) ...
        current_weights = self.get_dynamic_head_weights()
        return {
            'temporal': current_weights[0].item(),
            'spatial': current_weights[1].item(),
            'protocol': current_weights[2].item(),
            'statistical': current_weights[3].item(),
            'temperature': self.temperature.item()
        }
        # ... (no changes in this function) ...