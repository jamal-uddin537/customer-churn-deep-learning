"""Deep learning model: entity embeddings for categoricals + residual MLP."""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim, dim),
            nn.BatchNorm1d(dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ChurnNet(nn.Module):
    """Tabular network: each categorical column gets a learned embedding
    (size ~ min(16, (n+1)//2)), concatenated with scaled continuous features,
    then projected and passed through residual blocks to a single churn logit."""

    def __init__(self, cardinalities, n_cont, hidden=128, n_blocks=2, dropout=0.3, emb_dropout=0.1):
        super().__init__()
        self.embs = nn.ModuleList(
            nn.Embedding(n, min(16, (n + 1) // 2)) for n in cardinalities
        )
        emb_dim = sum(e.embedding_dim for e in self.embs)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.cont_bn = nn.BatchNorm1d(n_cont)
        self.proj = nn.Linear(emb_dim + n_cont, hidden)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.BatchNorm1d(hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x_cont, x_cat):
        e = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)
        x = torch.cat([self.emb_drop(e), self.cont_bn(x_cont)], dim=1)
        return self.head(self.blocks(self.proj(x))).squeeze(-1)
