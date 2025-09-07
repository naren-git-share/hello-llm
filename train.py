import torch
import torch.nn.functional as F
from model import TinyLLM

device = "cuda" if torch.cuda.is_available() else "cpu"
state_path = "tiny_llm_math_state.pth"

def train_model():
    model = TinyLLM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Simple sequences: [[1,3,5], [2,4,6], ...]
    sequences = [
        [1, 3, 5],
        [5, 10, 15],
        [2, 4, 6],
        [3, 6, 9]
    ]
    
    inputs = [seq[:-1] for seq in sequences]
    targets = [seq[1:] for seq in sequences]

    # Convert to tokens
    from model import NUMBER_TO_TOKEN
    input_tokens = torch.tensor([[NUMBER_TO_TOKEN[n] for n in seq] for seq in inputs], device=device)
    target_tokens = torch.tensor([[NUMBER_TO_TOKEN[n] for n in seq] for seq in targets], device=device)

    # Training loop
    for epoch in range(100):
        optimizer.zero_grad()
        logits = model(input_tokens)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Save weights
    torch.save(model.state_dict(), state_path)
    print(f"Model saved as {state_path}")

if __name__ == "__main__":
    train_model()