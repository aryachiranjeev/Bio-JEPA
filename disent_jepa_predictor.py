import torch
import torch.nn as nn

class JEPAPredictor(nn.Module):
    """
    Input : (B, N, 512)
    Output: (B, N, 512)
    """
    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        output_dim=512,
        depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        embed_dim = input_dim
        predictor_dim = hidden_dim
        output_dim = output_dim
        
        self.input_proj = nn.Linear(embed_dim, predictor_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=int(predictor_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.output_proj = nn.Linear(predictor_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        return x
    

class FusionPredictor(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        num_tokens=257, #196,      # N
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        # ---------------------------
        # Transformer over concatenated tokens
        # ---------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        # ---------------------------
        # Learnable latent queries
        # ---------------------------
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_tokens, embed_dim) * 0.02
        )

        # ---------------------------
        # Cross Attention
        # ---------------------------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, bio_pred, nonbio_pred):
        """
        bio_pred     : (B,N,D)
        nonbio_pred  : (B,N,D)
        """

        B = bio_pred.size(0)

        # ---------------------------------
        # Concatenate token sequences
        # (B,2N,D)
        # ---------------------------------
        memory = torch.cat([bio_pred, nonbio_pred], dim=1)

        # ---------------------------------
        # Global interaction
        # ---------------------------------
        memory = self.encoder(memory)

        # ---------------------------------
        # Expand latent queries
        # ---------------------------------
        queries = self.query_tokens.expand(B, -1, -1)

        # ---------------------------------
        # Cross-attention
        # Query : latent queries
        # Key   : memory
        # Value : memory
        # ---------------------------------
        output, attn = self.cross_attn(
            query=queries,
            key=memory,
            value=memory,
            need_weights=True,
            average_attn_weights=False,
        )

        output = self.norm(output)

        return output
    

# class FusionPredictor(nn.Module):
#     def __init__(self,
#                  input_dim=512,
#                  hidden_dim=1024,
#                  dropout=0.1):
#         super().__init__()
        
#         self.norm = nn.LayerNorm(input_dim * 2)
#         self.fusion = nn.Sequential(
#             nn.Linear(input_dim * 2, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, input_dim),
#         )

#     def forward(self, bio_pred, nonbio_pred):
#         x = torch.cat([bio_pred, nonbio_pred], dim=1) ## concat on tokens (B, 2N, 512)
#         x = self.norm(x)
#         x = self.fusion(x)
#         return x