# Biomedical Word Autocompletion System

## Overview
This project investigates next-word prediction on biomedical abstracts through a dual-model comparison:

1. HPYLM: a non-parametric Bayesian language model (Hierarchical Pitman-Yor Language Model) trained with Gibbs sampling.
2. RNN: a neural information bottleneck approach using recurrent dynamics to compress context and predict the next token.

The shared preprocessing pipeline standardizes tokenization, vocabulary truncation, and unknown-token handling so both model families can be evaluated under a consistent data regime.

## Repository Structure
```text
.
|-- data/
|   |-- raw/
|   \-- processed/
|-- src/
|   |-- data_pipeline.py
|   |-- hpylm.py
|   |-- rnn.py
|   \-- evaluate.py
|-- models/
|-- results/
|-- requirements.txt
|-- .gitignore
\-- README.md
```

Folder guide:
- data/raw/: unmodified dataset snapshots and exports downloaded from data sources.
- data/processed/: tokenized/serialized datasets and intermediate artifacts produced by preprocessing.
- src/: implementation code for data, model training, and evaluation.
- models/: saved model checkpoints, including PyTorch .pth weights and future HPYLM states.
- results/: experiment outputs such as perplexity/latency tables and summary metrics.

## Environment Setup
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the environment:
   - Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     .venv\Scripts\activate
     ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Ensure heavyweight/local paths are ignored in Git. Your .gitignore should include:
   - .venv/
   - data/
   - models/

## Usage
Run the evaluation entry point from the repository root:

```bash
python src/evaluate.py --vocab-size 10000 --seq-len 20 --batch-size 64
```

This script validates pipeline outputs for both HPYLM-oriented and RNN-oriented workflows, and reports dataset-window statistics for reproducible experimentation.
