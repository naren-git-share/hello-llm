from fastapi import FastAPI
import torch
import torch.nn.functional as F
from model import TinyLLM
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
state_path = "tiny_llm_math_state.pth"
vocab_path = "vocab.json"

# Load vocab
with open(vocab_path, "r") as f:
    vocab = json.load(f)

NUMBER_TO_TOKEN = {int(k): v for k, v in vocab["NUMBER_TO_TOKEN"].items()}
TOKEN_TO_NUMBER = {int(k): v for k, v in vocab["TOKEN_TO_NUMBER"].items()}

# Load model
model = TinyLLM(vocab_size=len(NUMBER_TO_TOKEN))
state_dict = torch.load(state_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

app = FastAPI()

@app.get("/predict")
def predict(numbers: str):
    nums = [int(n) for n in numbers.split(",")]
    try:
        tokens = [NUMBER_TO_TOKEN[n] for n in nums]
    except KeyError as e:
        return {"error": f"Number {e.args[0]} not in training vocab"}

    input_seq = torch.tensor([tokens], device=device)
    with torch.no_grad():
        logits = model(input_seq)
        next_token = torch.argmax(F.softmax(logits[:, -1, :], dim=-1), dim=-1)

    return {"next_number": TOKEN_TO_NUMBER[int(next_token.item())]}
