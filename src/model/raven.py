import torch
import torch.nn as nn
import torch.nn.functional as F

class RavenAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        return self.proj(x)

class RavenBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = RavenAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Raven(nn.Module):
    def __init__(self, vocab_size=32000, embed_dim=1024, num_layers=24, num_heads=16):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([RavenBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx, mask=None):
        x = self.token_emb(idx)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.ln_f(x))

if __name__ == "__main__":
    print("Initializing Raven...")
    model = Raven()
    test_input = torch.randint(0, 32000, (1, 128))
    output = model(test_input)
    print(f"Raven Inference Successful. Output shape: {output.shape}")
