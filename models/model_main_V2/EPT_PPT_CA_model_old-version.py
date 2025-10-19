import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class PacketImplicitTransformer(nn.Module):
    def __init__(self, config: Dict, packet_features_dim: int):
        super().__init__()

        embed_dim = config['model']['packet_embed_dim']
        protocol_dim = config['model']['protocol_embed_dim']

        self.packet_embedding = nn.Linear(packet_features_dim - 1, embed_dim)  # -1 for protocol
        self.protocol_embedding = nn.Embedding(20, protocol_dim)
        self.positional_encoding = PositionalEncoding(embed_dim + protocol_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim + protocol_dim,
            nhead=config['model']['packet_implicit']['num_heads'],
            dim_feedforward=config['model']['packet_implicit']['hidden_dim'],
            dropout=config['model']['packet_implicit']['dropout'],
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['model']['packet_implicit']['num_layers']
        )

        self.output_dim = embed_dim + protocol_dim

    def forward(self, packets: torch.Tensor, packet_masks: torch.Tensor = None):
        """
        Args:
            packets: [batch_size, flows_per_batch, max_packets, packet_features]
            packet_masks: [batch_size, flows_per_batch, max_packets] - 1 for real packets, 0 for padding
        """
        batch_size, flows_per_batch, max_packets, packet_features = packets.shape

        # Reshape for processing
        packets = packets.view(batch_size * flows_per_batch, max_packets, packet_features)

        if packet_masks is not None:
            packet_masks = packet_masks.view(batch_size * flows_per_batch, max_packets)

        # Extract protocol information (currently last feature)
        protocols = (packets[:, :, 5] * 10).long().clamp(0, 19)
        packet_features_only = torch.cat([packets[:, :, :5], packets[:, :, 6:]], dim=-1)  # exclude protocol

        # Embeddings
        packet_embeds = self.packet_embedding(packet_features_only)
        protocol_embeds = self.protocol_embedding(protocols)

        # Combine embeddings
        combined_embeds = torch.cat([packet_embeds, protocol_embeds], dim=-1)
        combined_embeds = self.positional_encoding(combined_embeds.transpose(0, 1)).transpose(0, 1)

        # Create attention mask for transformer (True for positions to ignore)
        if packet_masks is not None:
            attn_mask = ~packet_masks.bool()
        else:
            packet_sum = packets.sum(dim=-1)
            attn_mask = (packet_sum == 0)

        # Transform with attention mask
        transformed = self.transformer(combined_embeds, src_key_padding_mask=attn_mask)

        # Masked global average pooling over valid packets
        if packet_masks is not None:
            valid_mask = packet_masks.unsqueeze(-1).float()
            pooled = (transformed * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)
        else:
            valid_mask = (~attn_mask).unsqueeze(-1).float()
            pooled = (transformed * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)

        # Reshape back
        pooled = pooled.view(batch_size, flows_per_batch, self.output_dim)

        return pooled


class ExplicitAttentionHeads(nn.Module):
    """Four explicit attention heads using feature indices from preprocessed data"""

    def __init__(self, config: Dict, flow_features_dim: int, feature_info: Dict = None):
        super().__init__()

        self.config = config['model']['explicit_heads']
        head_dim = self.config['head_dim']
        num_heads = self.config['num_heads']

        # Use feature indices from preprocessing (the actual indices, not names)
        if feature_info and 'feature_indices' in feature_info:
            self.feature_ranges = feature_info['feature_indices']
            print(f"✓ Using feature indices from preprocessing: {self.feature_ranges}")
        else:
            # Fallback: divide features equally
            features_per_head = flow_features_dim // 4
            self.feature_ranges = {
                'temporal': [0, features_per_head],
                'spatial': [features_per_head, features_per_head * 2],
                'protocol': [features_per_head * 2, features_per_head * 3],
                'feature': [features_per_head * 3, flow_features_dim]
            }
            print(f"Using fallback feature mapping: {self.feature_ranges}")

        # Create attention heads for each group
        self.attention_heads = nn.ModuleDict()
        valid_heads = 0

        for head_name, feature_range in self.feature_ranges.items():
            if len(feature_range) != 2:
                print(f"Warning: Invalid feature range for {head_name}: {feature_range}")
                continue

            start_idx, end_idx = feature_range
            feature_dim = end_idx - start_idx

            if feature_dim <= 0:
                print(f"Warning: Empty feature group {head_name} ({start_idx}, {end_idx})")
                continue

            print(f"Creating attention head for {head_name}: features [{start_idx}:{end_idx}] (dim={feature_dim})")

            self.attention_heads[head_name] = nn.ModuleDict({
                'projection': nn.Linear(feature_dim, head_dim),
                'attention': nn.MultiheadAttention(
                    embed_dim=head_dim,
                    num_heads=min(num_heads, head_dim),
                    dropout=0.1,
                    batch_first=True
                ),
                'norm1': nn.LayerNorm(head_dim),
                'norm2': nn.LayerNorm(head_dim),
                'ffn': nn.Sequential(
                    nn.Linear(head_dim, head_dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(head_dim * 2, head_dim)
                )
            })
            valid_heads += 1

        if valid_heads == 0:
            print("Warning: No valid attention heads created!")
            self.output_dim = head_dim
        else:
            self.output_dim = head_dim * valid_heads

        print(f"Created {valid_heads} explicit attention heads, output dim: {self.output_dim}")

    def forward(self, flow_features: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            flow_features: [batch_size, flows_per_batch, flow_features_dim]
            mask: [batch_size, flows_per_batch]
        """
        batch_size, flows_per_batch = flow_features.shape[:2]

        head_outputs = []

        for head_name, feature_range in self.feature_ranges.items():
            if head_name not in self.attention_heads:
                continue

            # Extract feature group using actual indices
            start_idx, end_idx = feature_range
            if start_idx >= end_idx or end_idx > flow_features.shape[-1]:
                continue

            group_features = flow_features[:, :, start_idx:end_idx]

            # Reshape for attention
            group_features = group_features.view(batch_size * flows_per_batch, 1, -1)

            # Project to head dimension
            projected = self.attention_heads[head_name]['projection'](group_features)

            # Self-attention with residual connection
            attended, _ = self.attention_heads[head_name]['attention'](projected, projected, projected)
            attended = self.attention_heads[head_name]['norm1'](attended + projected)

            # Feed forward with residual connection
            ffn_output = self.attention_heads[head_name]['ffn'](attended)
            output = self.attention_heads[head_name]['norm2'](ffn_output + attended)

            # Reshape back
            output = output.view(batch_size, flows_per_batch, -1)
            head_outputs.append(output.squeeze(-2))

        if not head_outputs:
            dummy_output = torch.zeros(batch_size, flows_per_batch, self.config['head_dim'],
                                       device=flow_features.device)
            return dummy_output

        # Concatenate all head outputs
        return torch.cat(head_outputs, dim=-1)


class CrossFlowAttention(nn.Module):
    """Cross-attention mechanism across multiple flows with improved stability"""

    def __init__(self, config: Dict):
        super().__init__()

        # Transformer encoder for cross-flow attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,  # Input dimension after fusion
            nhead=config['model']['cross_attention']['num_heads'],
            dim_feedforward=config['model']['cross_attention']['hidden_dim'],
            dropout=config['model']['cross_attention']['dropout'],
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['model']['cross_attention']['num_layers']
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, 256))

        # Final projection
        self.final_projection = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, flow_representations: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            flow_representations: [batch_size, flows_per_batch, hidden_dim]
            mask: [batch_size, flows_per_batch]
        """
        batch_size, flows_per_batch, hidden_dim = flow_representations.shape

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, 1, -1)
        flow_representations = torch.cat([cls_tokens, flow_representations], dim=1)

        # Extend mask for CLS token
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
            extended_mask = torch.cat([cls_mask, mask], dim=1)
            attn_mask = ~extended_mask.bool()
        else:
            attn_mask = None

        # Cross-flow attention
        attended = self.transformer(flow_representations, src_key_padding_mask=attn_mask)

        # Get CLS token representation and apply final projection
        cls_output = attended[:, 0, :]  # [batch_size, hidden_dim]
        final_output = self.final_projection(cls_output)

        return final_output


class ExplicitPacketModel(nn.Module):
    """Model with only Explicit Flow Transformer + Packet Transformer + Cross Flow Attention"""

    def __init__(self, config: Dict, flow_features_dim: int, packet_features_dim: int,
                 feature_info: Dict = None):
        super().__init__()

        self.config = config

        print(f"Initializing Explicit+Packet Model with:")
        print(f"  Flow features dim: {flow_features_dim}")
        print(f"  Packet features dim: {packet_features_dim}")

        # Only packet and explicit transformers
        self.packet_transformer = PacketImplicitTransformer(config, packet_features_dim)
        self.explicit_heads = ExplicitAttentionHeads(config, flow_features_dim, feature_info)

        # Fusion layer (no implicit flow transformer)
        fusion_dim = (self.packet_transformer.output_dim + self.explicit_heads.output_dim)

        print(f"  Fusion dimensions: packet({self.packet_transformer.output_dim}) + "
              f"explicit({self.explicit_heads.output_dim}) = {fusion_dim}")

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        )

        # Cross-flow attention
        self.cross_flow_attention = CrossFlowAttention(config)

        # Classifier
        classifier_config = config['model']['classifier']
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_config['dropout']),
            nn.Linear(256, classifier_config['hidden_dim']),
            nn.LayerNorm(classifier_config['hidden_dim']),
            nn.GELU(),
            nn.Dropout(classifier_config['final_dropout']),
            nn.Linear(classifier_config['hidden_dim'], config['data']['num_classes'])
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with proper scaling"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, 0, 0.1)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            batch: Dictionary containing 'packets', 'flow_features', 'labels', 'mask', 'packet_masks'

        Returns:
            logits: [batch_size, num_classes]
            attention_weights: Dictionary of attention weights for interpretability
        """
        packets = batch['packets']
        flow_features = batch['flow_features']
        mask = batch['mask']
        packet_masks = batch.get('packet_masks')

        # Process through packet and explicit transformers only
        packet_repr = self.packet_transformer(packets, packet_masks)  # [batch_size, flows_per_batch, dim]
        explicit_repr = self.explicit_heads(flow_features, mask)  # [batch_size, flows_per_batch, dim]

        # Fusion (no implicit flow representation)
        combined_repr = torch.cat([packet_repr, explicit_repr], dim=-1)
        fused_repr = self.fusion_layer(combined_repr)  # [batch_size, flows_per_batch, 256]

        # Cross-flow attention
        final_repr = self.cross_flow_attention(fused_repr, mask)  # [batch_size, 256]

        # Classification
        logits = self.classifier(final_repr)  # [batch_size, num_classes]

        # Placeholder attention weights
        attention_weights = {
            'packet_attention': None,
            'explicit_attention': None,
            'cross_flow_attention': None
        }

        return logits, attention_weights

    def get_representation(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the final representation before classification (for fusion with implicit model)

        Args:
            batch: Dictionary containing 'packets', 'flow_features', 'labels', 'mask', 'packet_masks'

        Returns:
            final_repr: [batch_size, 256] - representation after cross-flow attention
        """
        packets = batch['packets']
        flow_features = batch['flow_features']
        mask = batch['mask']
        packet_masks = batch.get('packet_masks')

        # Process through packet and explicit transformers
        packet_repr = self.packet_transformer(packets, packet_masks)  # [batch_size, flows_per_batch, dim]
        explicit_repr = self.explicit_heads(flow_features, mask)  # [batch_size, flows_per_batch, dim]

        # Fusion of explicit + packet
        combined_repr = torch.cat([packet_repr, explicit_repr], dim=-1)
        fused_repr = self.fusion_layer(combined_repr)  # [batch_size, flows_per_batch, 256]

        # Cross-flow attention (this is where the sequences of N flows get processed)
        final_repr = self.cross_flow_attention(fused_repr, mask)  # [batch_size, 256]

        return final_repr