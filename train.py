import torch
import torch.nn.functional as F
from model import TinyLLM
import json
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
state_path = "tiny_llm_math_state.pth"
vocab_path = "vocab.json"
sequences_dir = Path("sequences")  # Folder containing your sequence files

def load_sequences():
    sequences = []
    for seq_file in sequences_dir.glob("*.txt"):
        with open(seq_file) as f:
            nums = [int(n) for n in f.read().strip().split(",")]
            sequences.append(nums)
    return sequences

def train_model():
    sequences = load_sequences()
    all_numbers = sorted(set(num for seq in sequences for num in seq))
    
    # Dynamic vocab
    NUMBER_TO_TOKEN = {n: i for i, n in enumerate(all_numbers)}
    TOKEN_TO_NUMBER = {i: n for n, i in NUMBER_TO_TOKEN.items()}

    model = TinyLLM(vocab_size=len(NUMBER_TO_TOKEN)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop: train one sequence at a time
    for epoch in range(100):
        total_loss = 0
        for seq in sequences:
            input_seq = torch.tensor([[NUMBER_TO_TOKEN[n] for n in seq[:-1]]], device=device)
            target_seq = torch.tensor([[NUMBER_TO_TOKEN[n] for n in seq[1:]]], device=device)

            optimizer.zero_grad()
            logits = model(input_seq)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            avg_loss = total_loss / len(sequences)
            print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")

    # Save model and vocab
    torch.save(model.state_dict(), state_path)
    with open(vocab_path, "w") as f:
        json.dump({"NUMBER_TO_TOKEN": NUMBER_TO_TOKEN, "TOKEN_TO_NUMBER": TOKEN_TO_NUMBER}, f)

    print(f"Model saved as {state_path}")
    print(f"Vocab saved as {vocab_path}")
    print(f"Loaded {len(sequences)} sequences with {len(all_numbers)} unique numbers.")

if __name__ == "__main__":
    train_model()