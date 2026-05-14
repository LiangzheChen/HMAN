# HMAN: Hierarchical Molecular Attention Network

This repository provides a cleaned and self-contained implementation of **HMAN**:  
**Hierarchical Molecular Attention Network for Few-Shot Molecular Property Prediction**.

HMAN is designed for few-shot molecular property prediction, where each molecular property task is formulated as a **2-way K-shot classification problem**. The model identifies task-specific key substructures through hierarchical attention and improves molecular property prediction by combining atom-level and molecule-level attention.




---

## Project Structure

```bash
HMAN/
│
├── main.py          # Entry point for training and evaluation
├── data.py          # Dataset loading, molecular graph construction, and few-shot sampling
├── model.py         # HMAN model, GNN encoder, attention modules, and trainer
├── util.py          # Utility functions for logging, saving, parameter counting, etc.
│
├── data/            # Dataset directory
│   ├── tox21/
│   ├── sider/
│   ├── muv/
│   └── toxcast/
│
└── results/         # Training logs and checkpoints
