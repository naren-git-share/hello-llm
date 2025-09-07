# Tiny LLM Math Predictor

## Overview
Tiny LLM Math Predictor is a minimal transformer-based language model trained to predict the next number in simple arithmetic sequences. The project demonstrates how to train, save, and serve a tiny LLM using PyTorch and FastAPI, with a clean separation between training, model definition, and serving.

## Repository Structure
- `model.py`            : TinyLLM class and token mappings
- `train.py`            : Training script
- `serve.py`            : FastAPI server for predictions
- `visualize.py`        : Visualize embeddings and attention
- `tiny_llm_math_state.pth`  : Saved model weights
- `README.md`

## Features
- Predict the next number in simple sequences like `1,3 -> 5` or `5,10 -> 15`
- Clean separation of model, training, and serving
- Safe weight loading with `weights_only=True` to avoid PyTorch pickle security warnings
- Optional visualization of embeddings and attention maps
- Lightweight and suitable for local experimentation

## Usage

### 1️⃣ Train the Model
```bash
python train.py
```
Trains the TinyLLM on simple arithmetic sequences.
Saves weights to `tiny_llm_math_state.pth`.

### 2️⃣ Host the Model via REST API
```bash
uvicorn serve:app --reload
```
Open your browser or use curl to test:
```
http://127.0.0.1:8000/predict?numbers=1,3
http://127.0.0.1:8000/predict?numbers=5,10
```
Returns JSON with predicted next number, e.g., `{"next_number": 5}`

### 3️⃣ Visualize Embeddings & Attention
```bash
python visualize.py
```
Prints embedding vectors for the input sequence and plots attention maps.

## Examples
- Input: `1,3` → Predicted next: `5`
- Input: `5,10` → Predicted next: `15`
- Input: `2,4` → Predicted next: `6`
- Input: `3,6` → Predicted next: `9`

## Requirements
- Python 3.12+
- PyTorch
- FastAPI
- Uvicorn
- Matplotlib (for visualization)

## Getting Started
1. Clone the repository:
```bash
git clone <your-repo-url>
cd tiny_llm_math
```
2. Create a Conda environment:
```bash
conda create -n PyTinyLLMMath python=3.12
conda activate PyTinyLLMMath
```
3. Install dependencies:
```bash
pip install torch fastapi uvicorn matplotlib
```
4. Train the model:
```bash
python train.py
```
5. Start the server:
```bash
uvicorn serve:app --reload
```
6. Test predictions via browser or curl.  
http://127.0.0.1:8000/predict?numbers=1,2
## Notes
- Always train before serving. The server expects `tiny_llm_math_state.pth`.
- Visualization is optional and purely for inspecting embeddings and attention weights.
- Lightweight model designed for learning and experimentation, not production-scale LLM.

