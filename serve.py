from fastapi import FastAPI
import torch
import torch.nn.functional as F
from model import TinyLLM, NUMBER_TO_TOKEN, TOKEN_TO_NUMBER

device = "cuda" if torch.cuda.is_available() else "cpu"
state_path = "tiny_llm_math_state.pth"

# Load model
model = TinyLLM()
state_dict = torch.load(state_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

app = FastAPI()

@app.get("/predict")
def predict(numbers: str):
    nums = [int(n) for n in numbers.split(",")]
    tokens = [NUMBER_TO_TOKEN[n] for n in nums]
    input_seq = torch.tensor([tokens], device=device)

    with torch.no_grad():
        logits = model(input_seq)
        next_token = torch.argmax(F.softmax(logits[:, -1, :], dim=-1), dim=-1)

    return {"next_number": TOKEN_TO_NUMBER[int(next_token.item())]}
