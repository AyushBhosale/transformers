Transformer from Scratch: Reverse-Sort Task

A pure PyTorch implementation of the Transformer architecture built from the ground up to solve a sequence manipulation task. This project demonstrates a deep understanding of the inner workings of Attention mechanisms, Positional Encodings, and Encoder-Decoder stacks by implementing them manually rather than relying on high-level nn.Transformer abstractions. <br>
📌 Project Overview

The goal of this project is to train a Transformer to perform two simultaneous algorithmic tasks on a sequence of digits:

    Sort the sequence in ascending order.

    Reverse the original sequence.

The model takes a sequence of 7 random digits (0-9) and predicts a concatenated sequence of 14 digits (7 sorted + 7 reversed).
🛠 Architecture Built from Scratch

Unlike standard implementations, this project manually defines the core components to ensure full transparency and control over the architecture:

    Multi-Head Attention: Implemented Q, K, V matrix projections and the scaled dot-product attention mechanism manually.

    Positional Encoding: Includes both a loop-based implementation (for mathematical clarity) and a vectorized implementation (for efficiency).

    Encoder/Decoder Blocks: Custom classes for stacking layers with residual connections and LayerNorm.

    Custom Training Loop: A manual training loop handling gradient zeroing, backpropagation, and loss accumulation without high-level trainer wrappers.

Model Specs

    Input Sequence Length: 7

    Output Sequence Length: 14

    Embedding Dimension: 12

    Heads: 1 (scaled for small-data sorting task)

    Layers: 7 Encoders + 7 Decoders

📂 File Structure
Bash

reverse_sort_transformer.py  # The main script containing model classes, data generation, and training loop

🚀 Getting Started
Prerequisites

    Python 3.8+

    PyTorch

    NumPy (implicit in PyTorch usage)

Installation
Bash

pip install torch

Running the Model

The script is self-contained. It generates its own synthetic dataset, builds the model, and runs the training loop.
Bash

python reverse_sort_transformer.py

🧠 Dataset Details

The dataset is synthetically generated using torch.randint:

    Input (X): Random sequences of digits 0-9. Shape: (Batch, 64, 7).

    Target (Y): Concatenation of the sorted input and the reversed input. Shape: (Batch, 64, 14).
