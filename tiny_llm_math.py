import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Config -----
vocab_size = 20          # numbers 1..20
seq_len = 2              # input sequence length
hidden_dim = 64
num_heads = 2
ff_dim = 128
num_layers = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Transformer Block -----
class TinyTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x

# ----- Tiny LLM -----
class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = nn.ModuleList([TinyTransformerBlock() for _ in range(num_layers)])
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        logits = self.fc_out(x)
        return logits

# ----- Token Mapping -----
number_to_token = {i: i-1 for i in range(1, vocab_size+1)}
token_to_number = {v:k for k,v in number_to_token.items()}

# ----- Generate Math Sequence Dataset -----
sequences = []
targets = []

for start in range(1, 15):
    for step in [1,2,3,5]:
        seq = [start, start+step]
        target = start + 2*step
        if target <= vocab_size:
            sequences.append(seq)
            targets.append(target)

data = torch.tensor([[number_to_token[n] for n in seq] for seq in sequences], device=device)
targets = torch.tensor([number_to_token[t] for t in targets], device=device)

# ----- Training -----
model = TinyLLM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    optimizer.zero_grad()
    logits = model(data)
    logits_last = logits[:, -1, :]       # only last token prediction
    loss = loss_fn(logits_last, targets)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ----- Save Model -----
torch.save(model.state_dict(), "tiny_llm_math_state.pth")
print("Model saved as tiny_llm_math_state.pth")
