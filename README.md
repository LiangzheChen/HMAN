# HMAN: Hierarchical Molecular Attention Network

This repository provides a cleaned and self-contained implementation of **HMAN**:  
**Hierarchical Molecular Attention Network for Few-Shot Molecular Property Prediction**.

HMAN is designed for few-shot molecular property prediction, where each molecular property task is formulated as a **2-way K-shot binary classification problem**. The model identifies task-specific key substructures through hierarchical attention and improves molecular property prediction by combining atom-level and molecule-level attention.

---

## Overview

Existing few-shot molecular property prediction methods often treat all atoms in a molecule as equally important. However, different molecular properties may be determined by different task-specific substructures.

HMAN addresses this issue by introducing:

1. **GNN-based molecular feature extraction**
   - A molecular graph encoder extracts atomic features.
   - Molecular-level representations are obtained through graph pooling.
   - Class prototypes are computed from support samples.

2. **Atom-level attention**
   - The model uses atomic features, molecular features, and prototype features to compute atom importance.
   - The top-B atoms with the highest attention scores are selected to form key substructures.

3. **Molecule-level attention**
   - The selected substructures are used as query features.
   - Molecular features are used as key-value representations.
   - The top-k substructures are selected and integrated into the final molecular representation.

4. **Weighted negative log-likelihood loss**
   - The attention-derived weight distribution is optimized according to the prediction reward.
   - The final support loss is composed of classification loss and weighted negative log-likelihood loss.

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