"""Small decoder trained with a tokenizer and weights initialized locally."""

import torch
from torch import nn
from torch.nn import functional as F


class DecoderBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.attention_out = nn.Linear(width, width, bias=False)
        self.mlp = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 3), nn.GELU(),
            nn.Linear(width * 3, width),
        )

    def forward(self, x):
        batch, steps, width = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).split(width, dim=-1)
        shape = (batch, steps, self.heads, width // self.heads)
        q, k, v = (t.reshape(shape).transpose(1, 2) for t in (q, k, v))
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.attention_out(attended.transpose(1, 2).reshape(batch, steps, width))
        return x + self.mlp(x)


class PeriodModel(nn.Module):
    def __init__(self, vocab_size: int, context: int, width: int, heads: int, layers: int):
        super().__init__()
        self.context = context
        self.tokens = nn.Embedding(vocab_size, width)
        self.positions = nn.Embedding(context, width)
        self.blocks = nn.Sequential(*(DecoderBlock(width, heads) for _ in range(layers)))
        self.norm = nn.LayerNorm(width)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.tokens(tokens) + self.positions(positions)
        return F.linear(self.norm(self.blocks(x)), self.tokens.weight)

    @torch.inference_mode()
    def complete(self, prefix: list[int], eos: int) -> list[int] | None:
        tokens = torch.tensor([prefix], device=self.tokens.weight.device)
        for _ in range(self.context - len(prefix)):
            next_token = self(tokens)[:, -1].argmax(dim=-1, keepdim=True)
            if next_token.item() == eos:
                return tokens[0, len(prefix):].tolist()
            tokens = torch.cat((tokens, next_token), dim=1)
        return None
