# Tiny LLM Math Sequence – End-to-End Local Python Demo

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)

**A minimal, GPU-accelerated, local LLM built in Python to predict simple arithmetic sequences.**  
This project demonstrates the **full lifecycle of a tiny transformer LLM**: training, hosting via FastAPI, and visualizing embeddings and attention maps — all on a single machine.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [How It Works](#how-it-works)  
- [Visualizations](#visualizations)  
- [Learning Outcomes](#learning-outcomes)  
- [Keywords](#keywords)  

---

## Project Overview

This project is designed for developers and data enthusiasts who want to **understand transformer-based LLMs locally**, without relying on cloud APIs or large-scale models.  

The model predicts the **next number in simple arithmetic sequences**.  
Input: 1, 3 → Predicted next: 5.  
Input: 5, 10 → Predicted next: 15.  

It demonstrates **end-to-end LLM workflow**:  

1. Training a tiny transformer from scratch.  
2. Saving weights safely using `state_dict`.  
3. Hosting via FastAPI for local REST API queries.  
4. Visualizing hidden embeddings and attention maps.  

---

## Features

- Minimal LLM with **single transformer block** and multi-head attention.  
- **State_dict-based model saving** for safe loading.  
- GPU acceleration via PyTorch + CUDA (optional).  
- REST API hosting with FastAPI — query via browser.  
- Interactive visualization of embeddings and attention maps.  
- Easy-to-extend for more complex sequences or operations.  

---

## Installation

> Requires [Conda](https://docs.conda.io/en/latest/) and Python 3.12.11+  

```bash
# 1. Clone repository
git clone https://github.com/yourusername/tiny-llm-math.git
cd tiny-llm-math

# 2. Create Conda environment
conda create -n PyTinyLLMMath python=3.12.11 -y
conda activate PyTinyLLMMath

# 3. Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. Install additional dependencies
pip install fastapi uvicorn matplotlib
