"""
Cross-Attention Block for fusing ReID embeddings with LLM hidden states.

Query:  ReID embedding     (B, 512) or (B, S_reid, 512)
Key/Value: LLM hidden states  (B, seq_length, 3584)
Mask:   Binary mask            (B, seq_length) — 1 = valid, 0 = padding

Output: Fused embedding        (B, output_dim)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlockFacial(nn.Module):
    """
    Cross-attention that uses ReID embeddings as queries and
    LLM hidden states as keys/values.

    Args:
        reid_dim:       Dimension of ReID embeddings (default: 512)
        llm_dim:        Dimension of LLM hidden states (default: 3584)
        proj_dim:       Internal projection dimension for attention (default: 512)
        num_heads:      Number of attention heads (default: 8)
        output_dim:     Final output dimension (default: 512)
        dropout:        Dropout rate (default: 0.1)
        num_layers:     Number of stacked cross-attention layers (default: 1)
    """

    def __init__(
        self,
        reid_dim: int = 1024, #512,
        llm_dim: int = 3584,
        proj_dim: int = 512,
        num_heads: int = 4, #8,
        output_dim: int = 512,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()
        assert proj_dim % num_heads == 0, f"proj_dim ({proj_dim}) must be divisible by num_heads ({num_heads})"

        self.reid_dim = reid_dim
        self.llm_dim = llm_dim
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_layers = num_layers

        # -----------Per-layer parameters-------------
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                query_dim=proj_dim, #reid_dim if i == 0 else proj_dim, ##chiranjeev - 8 july
                kv_dim=llm_dim,
                proj_dim=proj_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        # ----------- Input projection for ReID (if dim mismatch with proj_dim) -----------
        self.reid_input_proj = (
            nn.Linear(reid_dim, proj_dim)
            if reid_dim != proj_dim and num_layers > 1
            else nn.Identity()
        )

        # ----------- Output head -----------
        self.output_proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        reid_embed: torch.Tensor,
        llm_hidden: torch.Tensor,
        mask: torch.Tensor = None,
        bio = True,
    ) -> torch.Tensor:
        """
        Args:
            reid_embed: (B, 512) or (B, S_reid, 512)
            llm_hidden: (B, seq_length, 3584)
            mask:       (B, seq_length) — 1 = valid token, 0 = padding
                        Optional. If None, all positions are attended to.

        Returns:
            output: (B, output_dim) — fused embedding
        """
        # ----------- Handle 2D reid input: (B, 512) → (B, 1, 512) -----------
        squeeze_output = False
        if reid_embed.dim() == 2:
            reid_embed = reid_embed.unsqueeze(1)  # (B, 1, 512)
            squeeze_output = True

        # ----------- Build attention mask -----------
        # mask: (B, seq_length) → (B, 1, 1, seq_length) for broadcasting
        # 1 = attend, 0 = mask out
        attn_mask = None
        if mask is not None:
            # Convert 0/1 mask to additive mask: 0 → 0.0, 1 → -inf
            attn_mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, S_kv)
            attn_mask = (1.0 - attn_mask.float()) * (-1e9)

        # ----------- Project ReID for residual in multi-layer case -----------
        x = reid_embed
        x = self.reid_input_proj(x) ##chiranjeev - 8 july
        # if self.num_layers > 1: ##chiranjeev - 8 july
        #     x = self.reid_input_proj(x) ##chiranjeev - 8 july

        # ----------- Cross-attention layers -----------
        for i, layer in enumerate(self.layers):
            # query_input = reid_embed if i == 0 else x  ##chiranjeev - 8 july
            query_input = x ##chiranjeev - 8 july
            x, attn_weights = layer(query_input, llm_hidden, attn_mask, residual=x if i > 0 else None) ##chiranjeev - 8 july
            # x = layer(query_input, llm_hidden, attn_mask, residual=x if i > 0 else None) ##chiranjeev - 8 july
        
        if bio==False:
            query_input = query_input.detach()  # Detach query_input for non-bio branch to avoid gradient flow ##chiranjeev - 8 july
        fm_reid_proj_query_embed = ((1-attn_weights.mean(dim=1)).mean(dim=-1)).unsqueeze(-1)*query_input  # (B, S_q, proj_dim) #(1-attn_weights.mean(dim=1)).mean(dim=-1): (b,S_q)


        # ----------- Pool across query tokens → (B, proj_dim) -----------
        # if squeeze_output:
        #     x = x.squeeze(1)                    # (B, proj_dim)
        # else:
        #     x = x.mean(dim=1)                   # (B, S_reid, proj_dim) → (B, proj_dim)

        # ----------- Output projection -----------
        output = self.output_norm(x)        # (B, output_dim) ##18-june
        output = self.output_proj(x)             # (B, output_dim)
        # output = self.output_norm(output)        # (B, output_dim) ##earlier before 18 june

        return output, fm_reid_proj_query_embed


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer with pre-norm and residual.

    Query:     from ReID embeddings
    Key/Value: from LLM hidden states
    """

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        proj_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.proj_dim = proj_dim

        # ----------- Pre-norms -----------
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)

        # ----------- Projections -----------
        self.W_q = nn.Linear(query_dim, proj_dim, bias=False)
        self.W_k = nn.Linear(kv_dim, proj_dim, bias=False)
        self.W_v = nn.Linear(kv_dim, proj_dim, bias=False)
        self.W_o = nn.Linear(proj_dim, proj_dim)

        # ----------- Feed-forward -----------
        self.ff_norm = nn.LayerNorm(proj_dim)
        self.ff = nn.Sequential(
            nn.Linear(proj_dim, proj_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 4, proj_dim),
            nn.Dropout(dropout),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # ----------- Residual projection (if query_dim != proj_dim) -----------
        self.residual_proj = (
            nn.Linear(query_dim, proj_dim, bias=False)
            if query_dim != proj_dim
            else nn.Identity()
        )

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        attn_mask: torch.Tensor = None,
        residual: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            query:     (B, S_q, query_dim)
            kv:        (B, S_kv, kv_dim)
            attn_mask: (B, 1, 1, S_kv) additive mask (-1e9 for masked positions)
            residual:  (B, S_q, proj_dim) optional residual from previous layer

        Returns:
            (B, S_q, proj_dim)
        """
        B, S_q, _ = query.shape
        S_kv = kv.shape[1]

        # ----------- Residual -----------
        if residual is not None:
            res = residual
        else:
            res = self.residual_proj(query)

        # ----------- Pre-norm -----------
        q = self.norm_q(query)
        k_v = self.norm_kv(kv)

        # ----------- Q, K, V projections -----------
        Q = self.W_q(q).view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)    # (B, H, S_q, d)
        K = self.W_k(k_v).view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, S_kv, d)
        V = self.W_v(k_v).view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, S_kv, d)

        # ----------- Scaled dot-product attention -----------
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, S_q, S_kv)

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask  # mask out padding

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # (B, H, S_q, d)

        # ----------- Merge heads -----------
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S_q, self.proj_dim)
        attn_output = self.W_o(attn_output)
        attn_output = self.proj_dropout(attn_output)

        # ----------- Residual + cross-attention -----------
        x = res + attn_output

        # ----------- Feed-forward with residual -----------
        x = x + self.ff(self.ff_norm(x))
        
        return x, attn_weights  ##chiranjeev - 8 july
        # return x ##chiranjeev - 8 july


# ----------- Usage Example -----------
'''
if __name__ == "__main__":
    B = 4
    seq_len = 128
    reid_dim = 512
    llm_dim = 3584

    model = CrossAttentionBlockFacial(
        reid_dim=reid_dim,
        llm_dim=llm_dim,
        proj_dim=512,
        num_heads=8,
        output_dim=512,
        dropout=0.1,
        num_layers=2,
    )

    # ----------- Test 1: ReID as (B, 512) — single vector per sample -----------
    reid_embed = torch.randn(B, reid_dim)
    llm_hidden = torch.randn(B, seq_len, llm_dim)
    mask = torch.ones(B, seq_len)
    mask[:, -20:] = 0  # last 20 tokens are padding

    output = model(reid_embed, llm_hidden, mask)
    print(f"Input:  reid={reid_embed.shape}, llm={llm_hidden.shape}, mask={mask.shape}")
    print(f"Output: {output.shape}")  # (4, 512)

    # ----------- Test 2: ReID as (B, S, 512) — sequence of patches -----------
    reid_embed_seq = torch.randn(B, 5, reid_dim)

    output_seq = model(reid_embed_seq, llm_hidden, mask)
    print(f"\nInput:  reid={reid_embed_seq.shape}, llm={llm_hidden.shape}")
    print(f"Output: {output_seq.shape}")  # (4, 512)

    # ----------- Test 3: No mask -----------
    output_no_mask = model(reid_embed, llm_hidden)
    print(f"\nNo mask output: {output_no_mask.shape}")  # (4, 512)

    # ----------- Parameter count -----------
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParams: {total:,} total, {trainable:,} trainable")

'''