import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaLiteBlock(nn.Module):
    """
    Practical Mamba-style sequence block.

    It keeps the important modelling idea for this project: gated long-sequence
    state mixing without Transformer attention. If the official mamba-ssm
    package is available, this project can be extended to replace this block,
    but this version runs with plain PyTorch.
    """

    def __init__(self, dim, expansion=2, kernel_size=5, dropout=0.1):
        super().__init__()
        inner = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, inner * 2)
        self.depthwise_conv = nn.Conv1d(
            inner,
            inner,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inner,
        )
        self.state_gate = nn.Sequential(
            nn.Linear(inner, inner),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        value = value.transpose(1, 2)
        value = self.depthwise_conv(value).transpose(1, 2)
        value = F.silu(value)

        gate = torch.sigmoid(gate)
        state = torch.cumsum(value * self.state_gate(value), dim=1)
        state = state / torch.arange(1, x.size(1) + 1, device=x.device).view(1, -1, 1)

        out = self.out_proj((value + state) * gate)
        return residual + self.dropout(out)


class CNNMambaClassifier(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        hidden_dim=128,
        num_blocks=3,
        dropout=0.2,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [MambaLiteBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, x):
        # x shape: batch, length, channels
        x = x.transpose(1, 2)
        x = self.cnn(x).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        avg_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        return torch.cat([avg_pool, max_pool], dim=-1)

    def forward(self, x):
        embedding = self.encode(x)
        logits = self.classifier(embedding)
        return logits
