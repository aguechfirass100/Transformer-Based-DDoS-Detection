# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class TemporalAttention(nn.Module):
#     def __init__(self, input_dim, embed_dim, dropout=0.1):
#         super().__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(input_dim, embed_dim * 2),
#             nn.ReLU(),
#             nn.LayerNorm(embed_dim * 2),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim * 2, embed_dim),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         # x shape: (batch_size, num_temporal_features)
#         return self.attention(x)

import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        return self.attention(x)