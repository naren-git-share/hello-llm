import torch
import torch.nn as nn

# Tiny Transformer LLM for simple math sequences
class TinyLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, num_heads=2, ff_hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, embed_dim)
        )
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embed(x)        # [batch, seq_len, embed_dim]
        x = x.transpose(0, 1)    # [seq_len, batch, embed_dim] for MultiheadAttention
        attn_out, _ = self.attn(x, x, x)
        x = attn_out + x
        x = self.ff(x) + x
        x = x.transpose(0, 1)    # back to [batch, seq_len, embed_dim]
        logits = self.out(x)
        return logits