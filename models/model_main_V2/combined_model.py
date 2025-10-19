import torch
import torch.nn as nn
from typing import Dict, Tuple

from EPT_PPT_CA_model import ExplicitPacketModel


class CombinedDDoSModel(nn.Module):
    """Final model that combines pre-trained implicit flow model with explicit+packet model"""

    def __init__(self, config: Dict, flow_features_dim: int, packet_features_dim: int,
                 implicit_model_path: str = None, explicit_packet_model_path: str = None,
                 feature_info: Dict = None):
        super().__init__()

        self.config = config

        # Load your EXISTING trained implicit flow model
        self.implicit_flow_model = None
        self.implicit_output_dim = 256  # You need to tell me what dimension your model outputs

        if implicit_model_path:
            print(f"Loading YOUR pre-trained implicit flow model from {implicit_model_path}")

            # Import your implicit model
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model_basic'))
            from models.model_basic.model import TransformerClassifier

            # Load your implicit model config (you'll need to provide this)
            implicit_config = config.get('implicit_model_config', config)  # fallback to main config if not specified

            # Create and load the model
            self.implicit_flow_model = TransformerClassifier(implicit_config)
            checkpoint = torch.load(implicit_model_path, map_location='cpu')
            self.implicit_flow_model.load_state_dict(checkpoint['state_dict'])

            # Set to eval mode
            self.implicit_flow_model.eval()

            # The output dimension should be embed_dim + protocol_embed_dim (before final classifier)
            embed_dim = implicit_config['model']['embed_dim']
            protocol_embed_dim = implicit_config['model'].get('protocol_embed_dim', embed_dim)  # fallback to embed_dim
            self.implicit_output_dim = embed_dim + protocol_embed_dim

            print(f"Implicit model output dimension: {self.implicit_output_dim}")

            # Freeze implicit model weights
            for param in self.implicit_flow_model.parameters():
                param.requires_grad = False
            print("✓ Implicit flow model loaded and frozen")

        # Load the explicit+packet model (will be trained separately)
        self.explicit_packet_model = ExplicitPacketModel(config, flow_features_dim, packet_features_dim, feature_info)
        if explicit_packet_model_path:
            print(f"Loading explicit+packet model from {explicit_packet_model_path}")
            explicit_checkpoint = torch.load(explicit_packet_model_path, map_location='cpu')
            self.explicit_packet_model.load_state_dict(explicit_checkpoint['state_dict'])
            # Freeze explicit+packet model weights initially
            for param in self.explicit_packet_model.parameters():
                param.requires_grad = False
            print("✓ Explicit+packet model loaded and frozen")

        # Final fusion and classification layers
        # You need to tell me the output dimension of your implicit model
        fusion_input_dim = self.implicit_output_dim + 256  # 256 from explicit+packet model

        print(
            f"Final fusion dimensions: implicit({self.implicit_output_dim}) + explicit+packet(256) = {fusion_input_dim}")

        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Final classifier
        self.final_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, config['data']['num_classes'])
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for new layers only"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def unfreeze_explicit_packet_model(self):
        """Unfreeze explicit+packet model for fine-tuning"""
        for param in self.explicit_packet_model.parameters():
            param.requires_grad = True
        print("✓ Explicit+packet model unfrozen for fine-tuning")

    def unfreeze_implicit_model(self):
        """Unfreeze implicit model for fine-tuning (use cautiously)"""
        for param in self.implicit_flow_model.parameters():
            param.requires_grad = True
        print("✓ Implicit flow model unfrozen for fine-tuning")

    def extract_protocol_ids_from_flow_features(self, flow_features: torch.Tensor) -> torch.Tensor:
        """
        Extract protocol IDs from flow features

        Args:
            flow_features: [batch_size, flows_per_batch, feature_dim]

        Returns:
            protocol_ids: [batch_size * flows_per_batch] - protocol IDs for each flow
        """
        batch_size, flows_per_batch, feature_dim = flow_features.shape

        # You need to identify which column/index contains the protocol information
        # This depends on your feature preprocessing - check your feature_info or config

        # Option 1: If protocol is at a specific index (you need to find this)
        protocol_feature_idx = 5  # UPDATE THIS - find the actual index of protocol in your features
        protocol_values = flow_features[:, :, protocol_feature_idx]  # [batch_size, flows_per_batch]

        # Option 2: If you know the feature name, you can use feature_info
        # if self.feature_info and 'feature_columns' in self.feature_info:
        #     feature_columns = self.feature_info['feature_columns']
        #     if 'Protocol' in feature_columns:
        #         protocol_idx = feature_columns.index('Protocol')
        #         protocol_values = flow_features[:, :, protocol_idx]

        # Convert to integer protocol IDs and flatten
        protocol_ids = (protocol_values * 10).long().clamp(0, 17)  # Adjust based on your protocol encoding
        protocol_ids = protocol_ids.view(-1)  # [batch_size * flows_per_batch]

        return protocol_ids

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            batch: Dictionary containing 'packets', 'flow_features', 'labels', 'mask', 'packet_masks'

        Returns:
            logits: [batch_size, num_classes]
            attention_weights: Dictionary of attention weights
        """
        flow_features = batch['flow_features']
        mask = batch['mask']

        # 1. Get implicit flow representation (frozen, pre-trained)
        with torch.no_grad() if not any(
                p.requires_grad for p in self.implicit_flow_model.parameters()) else torch.enable_grad():
            batch_size, flows_per_batch, feature_dim = flow_features.shape

            # Extract protocol IDs
            protocol_ids = self.extract_protocol_ids_from_flow_features(flow_features)

            # Reshape flow features for implicit model: [batch_size * flows_per_batch, feature_dim]
            flow_features_flat = flow_features.view(batch_size * flows_per_batch, feature_dim)

            # Get representation from implicit model (before classification)
            implicit_repr_flat = self.implicit_flow_model.get_representation(flow_features_flat, protocol_ids)
            # [batch_size * flows_per_batch, implicit_output_dim]

            # Reshape back and average across flows
            implicit_repr = implicit_repr_flat.view(batch_size, flows_per_batch,
                                                    -1)  # [batch_size, flows_per_batch, implicit_output_dim]
            implicit_repr = implicit_repr.mean(dim=1)  # [batch_size, implicit_output_dim]

        # 2. Get explicit+packet representation using the new method
        explicit_packet_repr = self.explicit_packet_model.get_representation(batch)  # [batch_size, 256]

        # 3. Final fusion of both representations
        final_combined = torch.cat([implicit_repr, explicit_packet_repr],
                                   dim=-1)  # [batch_size, implicit_output_dim + 256]
        final_fused = self.final_fusion(final_combined)  # [batch_size, 256]

        # 4. Final classification
        logits = self.final_classifier(final_fused)  # [batch_size, num_classes]

        # Placeholder attention weights (you can get real ones from explicit_packet_model if needed)
        attention_weights = {
            'implicit_attention': None,
            'explicit_packet_attention': None
        }

        return logits, attention_weights


class CombinedTrainingStrategy:
    """Updated training strategy that reflects the correct architecture"""

    def __init__(self, model: CombinedDDoSModel):
        self.model = model

    def phase1_train_fusion_only(self):
        """Phase 1: Train only the final fusion and classification layers"""
        # Freeze both pre-trained models
        if self.model.implicit_flow_model:
            for param in self.model.implicit_flow_model.parameters():
                param.requires_grad = False

        for param in self.model.explicit_packet_model.parameters():
            param.requires_grad = False

        # Only final fusion and classifier are trainable
        trainable_params = (list(self.model.final_fusion.parameters()) +
                            list(self.model.final_classifier.parameters()))
        print(f"Phase 1: Training {sum(p.numel() for p in trainable_params)} fusion parameters")
        print(
            "Architecture: [Frozen Implicit] + [Frozen Explicit+Packet+CrossAttn] -> [Trainable Fusion] -> [Trainable Classifier]")
        return trainable_params

    def phase2_finetune_explicit_packet(self):
        """Phase 2: Unfreeze explicit+packet model (including cross-attention) for fine-tuning"""
        self.model.unfreeze_explicit_packet_model()

        # Keep implicit model frozen, train explicit+packet+cross-attention + fusion
        trainable_params = (list(self.model.explicit_packet_model.parameters()) +
                            list(self.model.final_fusion.parameters()) +
                            list(self.model.final_classifier.parameters()))
        print(f"Phase 2: Training {sum(p.numel() for p in trainable_params)} parameters")
        print(
            "Architecture: [Frozen Implicit] + [Trainable Explicit+Packet+CrossAttn] -> [Trainable Fusion] -> [Trainable Classifier]")
        return trainable_params

    def phase3_finetune_all(self):
        """Phase 3: Unfreeze everything for full fine-tuning (use with extreme caution!)"""
        if self.model.implicit_flow_model:
            self.model.unfreeze_implicit_model()
        self.model.unfreeze_explicit_packet_model()

        trainable_params = list(self.model.parameters())
        print(f"Phase 3: Training {sum(p.numel() for p in trainable_params)} parameters (full model)")
        print(
            "Architecture: [Trainable Implicit] + [Trainable Explicit+Packet+CrossAttn] -> [Trainable Fusion] -> [Trainable Classifier]")
        print("WARNING: This may degrade the performance of your well-trained implicit model!")
        return trainable_params