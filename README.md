# Transformer from Scratch: Reverse–Sort Task

A **pure PyTorch implementation** of the Transformer architecture built entirely from first principles to solve a **sequence manipulation problem**.  
This project demonstrates a deep understanding of **self-attention**, **positional encodings**, and **encoder–decoder stacks** by avoiding high-level abstractions such as `nn.Transformer`.

---

## 📌 Project Overview

The objective of this project is to train a Transformer model to perform **two algorithmic tasks simultaneously** on a sequence of digits:

1. **Sort** the sequence in ascending order  
2. **Reverse** the original sequence  

### 🔢 Task Definition

- **Input:** A sequence of **7 random digits** (0–9)  
- **Output:** A concatenated sequence of **14 digits**  
  - First 7 → sorted sequence  
  - Next 7 → reversed sequence  

---

## 🛠 Architecture Built from Scratch

All core Transformer components are implemented manually to ensure transparency and architectural clarity.

### 🔹 Core Components

- **Multi-Head Attention**
  - Manual implementation of **Q, K, V projections**
  - Explicit **scaled dot-product attention**
- **Positional Encoding**
  - Loop-based implementation for mathematical clarity
  - Vectorized implementation for computational efficiency
- **Encoder / Decoder Blocks**
  - Custom stackable layers
  - Residual connections
  - Layer Normalization
- **Training Loop**
  - Fully manual loop
  - Explicit gradient zeroing
  - Backpropagation and loss tracking
  - No high-level training wrappers

---

## ⚙️ Model Specifications

| Parameter | Value |
|--------|-------|
| Input Sequence Length | 7 |
| Output Sequence Length | 14 |
| Embedding Dimension | 12 |
| Attention Heads | 1 |
| Encoder Layers | 7 |
| Decoder Layers | 7 |

*(Model scaled intentionally for small-data, algorithmic learning tasks.)*

---

## 📂 File Structure

```bash
reverse_sort_transformer.ipynb  # Model implementation, data generation, and training loop
```
## 🚀 Getting Started

### ✅ Prerequisites

Ensure the following dependencies are installed:

- **Python 3.8+**
- **PyTorch**
- **NumPy** (used implicitly via PyTorch)

---

### 📦 Installation

Install PyTorch using pip:

```bash
pip install torch
```

## ▶️ Running the Model

The script is **fully self-contained** and performs the following steps automatically:

- Generates a synthetic dataset  
- Builds the Transformer model from scratch  
- Executes the training loop  

Run the script using:

```bash
python reverse_sort_transformer.py
```

## 🧠 Dataset Details

The dataset is synthetically generated using `torch.randint`.

### 📥 Input (`X`)

- Random digit sequences in the range **0–9**
- Tensor shape:

```text
(Batch, 64, 7)
```

### 📤 Target (`Y`)

The target sequence is a concatenation of:

- The **sorted** input sequence  
- The **reversed** input sequence  

Tensor shape:

```text
(Batch, 64, 14)
```

