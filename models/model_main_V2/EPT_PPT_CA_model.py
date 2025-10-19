import torch
import torch.nn as nn
import math
import yaml
from typing import Dict, Tuple, Optional


from models.model_multi_attention.model2 import SimplifiedMultiAttentionTransformer


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


# class ExplicitPacketModel(nn.Module):
#     """Model with Simplified Multi-Attention Transformer + Packet Transformer + Cross Flow Attention"""
#
#     def __init__(self, config: Dict, flow_features_dim: int, packet_features_dim: int,
#                  feature_info: Dict = None):
#         super().__init__()
#
#         self.config = config
#
#         print(f"Initializing Explicit+Packet Model with Simplified Multi-Attention:")
#         print(f"  Flow features dim: {flow_features_dim}")
#         print(f"  Packet features dim: {packet_features_dim}")
#
#         # Packet transformer (unchanged)
#         self.packet_transformer = PacketImplicitTransformer(config, packet_features_dim)
#
#         multi_attention_config_file_path = r"C:\Users\AGFirass\Documents\GitHub\Transformer-Based-DDoS-Detection\models\model_multi_attention\config\config.yaml"
#         with open(multi_attention_config_file_path, 'r') as f:
#             multi_attention_config = yaml.safe_load(f)
#         # self.simplified_multi_attention = SimplifiedMultiAttentionTransformer(multi_attention_config)
#
#         self.simplified_multi_attention = SimplifiedMultiAttentionTransformer(config)
#
#         # Fusion layer - update dimensions
#         fusion_dim = (self.packet_transformer.output_dim + self.simplified_multi_attention.output_dim)
#
#         print(f"  Fusion dimensions: packet({self.packet_transformer.output_dim}) + "
#               f"simplified_multi_attention({self.simplified_multi_attention.output_dim}) = {fusion_dim}")
#
#         self.fusion_layer = nn.Sequential(
#             nn.Linear(fusion_dim, 512),
#             nn.LayerNorm(512),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256)
#         )
#
#         # Cross-flow attention (unchanged)
#         self.cross_flow_attention = CrossFlowAttention(config)
#
#         # Classifier (unchanged)
#         classifier_config = config['model']['classifier']
#         self.classifier = nn.Sequential(
#             nn.Dropout(classifier_config['dropout']),
#             nn.Linear(256, classifier_config['hidden_dim']),
#             nn.LayerNorm(classifier_config['hidden_dim']),
#             nn.GELU(),
#             nn.Dropout(classifier_config['final_dropout']),
#             nn.Linear(classifier_config['hidden_dim'], config['data']['num_classes'])
#         )
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, module):
#         """Initialize weights with proper scaling"""
#         if isinstance(module, nn.Linear):
#             torch.nn.init.xavier_uniform_(module.weight)
#             if module.bias is not None:
#                 module.bias.data.fill_(0.01)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, 0, 0.1)
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#
#     def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         """
#         Args:
#             batch: Dictionary containing 'packets', 'flow_features', 'labels', 'mask', 'packet_masks'
#
#         Returns:
#             logits: [batch_size, num_classes]
#             attention_weights: Dictionary of attention weights for interpretability
#         """
#
#
#         packets = batch['packets']
#         flow_features = batch['flow_features']
#         mask = batch['mask']
#         packet_masks = batch.get('packet_masks')
#
#         # Process through packet transformer
#         packet_repr = self.packet_transformer(packets, packet_masks)  # [batch_size, flows_per_batch, dim]
#
#         # Process through simplified multi-attention transformer
#         # multi_attention_repr = self.simplified_multi_attention(flow_features, mask)  # [batch_size, flows_per_batch, dim]
#         # multi_attention_repr = self.simplified_multi_attention(flow_features)  # [batch_size, flows_per_batch, dim]
#         multi_attention_repr = self.simplified_multi_attention(batch)  # [batch_size, flows_per_batch, dim]
#
#         # Fusion of packet + simplified multi-attention representations
#         combined_repr = torch.cat([packet_repr, multi_attention_repr], dim=-1)
#         fused_repr = self.fusion_layer(combined_repr)  # [batch_size, flows_per_batch, 256]
#
#         # Cross-flow attention
#         final_repr = self.cross_flow_attention(fused_repr, mask)  # [batch_size, 256]
#
#         # Classification
#         logits = self.classifier(final_repr)  # [batch_size, num_classes]
#
#         # Placeholder attention weights
#         attention_weights = {
#             'packet_attention': None,
#             'multi_attention': None,
#             'cross_flow_attention': None
#         }
#
#         return logits, attention_weights
#
#     def get_representation(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
#         """
#         Get the final representation before classification (for fusion with implicit model)
#
#         Args:
#             batch: Dictionary containing 'packets', 'flow_features', 'labels', 'mask', 'packet_masks'
#
#         Returns:
#             final_repr: [batch_size, 256] - representation after cross-flow attention
#         """
#         packets = batch['packets']
#         flow_features = batch['flow_features']
#         mask = batch['mask']
#         packet_masks = batch.get('packet_masks')
#
#         # Process through packet transformer
#         packet_repr = self.packet_transformer(packets, packet_masks)  # [batch_size, flows_per_batch, dim]
#
#         # Process through simplified multi-attention transformer
#         multi_attention_repr = self.simplified_multi_attention(flow_features,
#                                                                mask)  # [batch_size, flows_per_batch, dim]
#
#         # Fusion of packet + simplified multi-attention
#         combined_repr = torch.cat([packet_repr, multi_attention_repr], dim=-1)
#         fused_repr = self.fusion_layer(combined_repr)  # [batch_size, flows_per_batch, 256]
#
#         # Cross-flow attention (this is where the sequences of N flows get processed)
#         final_repr = self.cross_flow_attention(fused_repr, mask)  # [batch_size, 256]
#
#         return final_repr

class ExplicitPacketModel(nn.Module):
    """Explicit + Packet Hybrid Model with Learnable Weighted Fusion"""

    def __init__(self, config: Dict, flow_features_dim: int, packet_features_dim: int, feature_info: Dict = None):
        super().__init__()

        self.config = config

        print(f"Initializing Explicit+Packet Model with Learnable Weighted Fusion")
        print(f"  Flow features dim: {flow_features_dim}")
        print(f"  Packet features dim: {packet_features_dim}")

        # Packet transformer
        self.packet_transformer = PacketImplicitTransformer(config, packet_features_dim)

        # Multi-attention transformer
        self.simplified_multi_attention = SimplifiedMultiAttentionTransformer(config)

        # Project both streams to 256 dims
        self.packet_proj = nn.Linear(self.packet_transformer.output_dim, 256)
        self.flow_proj = nn.Linear(self.simplified_multi_attention.output_dim, 256)

        self.packet_norm = nn.LayerNorm(256)
        self.flow_norm = nn.LayerNorm(256)

        # Learnable scalar for weighted fusion (alpha for packet, 1-alpha for flow)
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
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
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, 0, 0.1)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, batch: Dict[str, torch.Tensor]):
        packets = batch['packets']
        flow_features = batch['flow_features']
        mask = batch['mask']
        packet_masks = batch.get('packet_masks')

        # --- Packet transformer ---
        packet_repr = self.packet_transformer(packets, packet_masks)  # [B, F, packet_dim]
        packet_repr = self.packet_proj(packet_repr)
        packet_repr = self.packet_norm(packet_repr)

        # --- Multi-attention transformer ---
        flow_repr = self.simplified_multi_attention(batch)  # [B, F, flow_dim]
        flow_repr = self.flow_proj(flow_repr)
        flow_repr = self.flow_norm(flow_repr)

        # --- Learnable weighted fusion ---
        combined = self.alpha * packet_repr + (1 - self.alpha) * flow_repr
        fused_repr = self.fusion_layer(combined)  # [B, F, 256]

        # --- Cross-flow attention ---
        final_repr = self.cross_flow_attention(fused_repr, mask)  # [B, 256]

        # --- Classification ---
        logits = self.classifier(final_repr)

        attention_weights = {
            'packet_attention': None,
            'multi_attention': None,
            'cross_flow_attention': None
        }

        return logits, attention_weights

    def get_representation(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get final representation before classification"""
        packets = batch['packets']
        mask = batch['mask']
        packet_masks = batch.get('packet_masks')

        packet_repr = self.packet_transformer(packets, packet_masks)
        packet_repr = self.packet_proj(packet_repr)
        packet_repr = self.packet_norm(packet_repr)

        flow_repr = self.simplified_multi_attention(batch)
        flow_repr = self.flow_proj(flow_repr)
        flow_repr = self.flow_norm(flow_repr)

        fused = self.fusion_layer(self.alpha * packet_repr + (1 - self.alpha) * flow_repr)
        final_repr = self.cross_flow_attention(fused, mask)
        return final_repr
