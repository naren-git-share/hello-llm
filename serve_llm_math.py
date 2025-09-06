from fastapi import FastAPI
import torch
import torch.nn.functional as F

from tiny_llm_math import TinyLLM  # import class definition

model = TinyLLM()
model.load_state_dict(torch.load("tiny_llm_math_state.pth", map_location="cuda" if torch.cuda.is_available() else "cpu"))
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

number_to_token = {i: i-1 for i in range(1, 21)}
token_to_number = {v:k for k,v in number_to_token.items()}

app = FastAPI()

@app.get("/predict")
def predict(numbers: str):
    nums = [int(n) for n in numbers.split(",")]
    tokens = [number_to_token[n] for n in nums]
    input_seq = torch.tensor([tokens], device=device)
    with torch.no_grad():
        logits = model(input_seq)
        next_token = torch.argmax(F.softmax(logits[:, -1, :], dim=-1), dim=-1)
    return {"next_number": token_to_number[int(next_token.item())]}
