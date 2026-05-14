# HMAN: Hierarchical Molecular Attention Network

This repository provides a cleaned and self-contained implementation of **HMAN**:  
**Hierarchical Molecular Attention Network for Few-Shot Molecular Property Prediction**.

HMAN is designed for few-shot molecular property prediction, where each molecular property task is formulated as a **2-way K-shot binary classification problem**. The model identifies task-specific key substructures through hierarchical attention and improves molecular property prediction by combining atom-level and molecule-level attention.



## Step-by-Step Running Workflow

The complete workflow includes five main steps:

```text
Model training → Model evaluation → Ablation analysis → Visualization
```

## 1. Environment  

We used the following Python packages for core development. We tested on `Python 3.7`.
```
- pytorch 1.7.0
- torch-geometric 1.7.0
```
---
## 2. Model Training

The main training procedure is implemented in `main.py`.

To train HMAN on a specific dataset, run:

```bash
python main.py --dataset tox21 --shot 10 
```

Example for another dataset:

```bash
python main.py --dataset sider --shot 10 
```

The trained model checkpoints and logs will be saved automatically.

## 3. Model Evaluation

To evaluate a trained HMAN model, run:

```bash
python main.py --dataset tox21 --shot 10 
```

The evaluation results will be printed in the console and saved in the result folder.

## 4. Ablation Analysis

To reproduce the ablation experiments, run:

```bash
python ablation.py --dataset tox21 --shot 10
```

This script evaluates the contribution of different components in HMAN, such as hierarchical attention and weighted loss design.

## 5. Visualization

To visualize the experimental results or the identified key substructures, run:

```bash
python visualization.py --dataset tox21 --shot 10
```

The generated figures will be saved in the visualization folder.

## Project Structure

```bash
HMAN/
│
├── main.py          # Entry point for training and evaluation
├── data.py          # Dataset loading, molecular graph construction, and few-shot sampling
├── model.py         # HMAN model, GNN encoder, attention modules, and trainer
├── util.py          # Utility functions for logging, saving, parameter counting, etc.
│── visualize.py     # Visualization script
├── data/            # Dataset directory
│   ├── tox21/
│   ├── sider/
│   ├── muv/
│   └── toxcast/
│
└── results/         # Training logs and checkpoints

