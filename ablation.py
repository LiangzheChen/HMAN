# ablation.py
"""
Ablation study script for HMAN.

This script evaluates four variants:
    1. full_hman
    2. w/o atom-level attention
    3. w/o molecule-level attention
    4. w/o weighted negative log-likelihood loss

It assumes the cleaned project structure:
    main.py
    data.py
    model.py
    util.py

Example:
    python ablation.py --dataset tox21 --data_dir ./data/ --epochs 5000 --gpu_id 0
"""

import argparse
import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from data import build_few_shot_tasks
from model import HMAN, HMANTrainer
from util import count_model_params, create_experiment_dir, save_config, set_logger, set_seed


class UniformAtomLevelAttention(nn.Module):
    """
    Ablation module for removing atom-level attention.

    Instead of selecting top-B task-specific atoms, this module uses the
    original molecular representation as the substructure representation.
    This keeps the interface unchanged while disabling atom-level selection.
    """

    def __init__(self, top_b_atoms: int = 5):
        super().__init__()
        self.top_b_atoms = top_b_atoms

    def forward(self, node_repr, graph_repr, prototypes, batch):
        substructure_repr = graph_repr

        atom_weight_per_graph = []
        selected_atom_indices = []
        selected_atom_weights = []

        num_graphs = graph_repr.size(0)

        for graph_id in range(num_graphs):
            atom_mask = batch == graph_id
            atom_indices = torch.nonzero(atom_mask, as_tuple=False).view(-1)
            num_atoms = atom_indices.numel()

            if num_atoms == 0:
                weights = torch.empty(0, device=graph_repr.device)
                selected_indices = torch.empty(0, dtype=torch.long, device=graph_repr.device)
                selected_weights = torch.empty(0, device=graph_repr.device)
            else:
                weights = torch.ones(num_atoms, device=graph_repr.device) / num_atoms
                k = min(self.top_b_atoms, num_atoms)
                selected_indices = atom_indices[:k]
                selected_weights = weights[:k]

            atom_weight_per_graph.append(weights)
            selected_atom_indices.append(selected_indices)
            selected_atom_weights.append(selected_weights)

        return {
            "substructure_repr": substructure_repr,
            "atom_weight_per_graph": atom_weight_per_graph,
            "selected_atom_indices": selected_atom_indices,
            "selected_atom_weights": selected_atom_weights,
        }


class IdentityMoleculeLevelAttention(nn.Module):
    """
    Ablation module for removing molecule-level attention.

    Instead of selecting top-k substructures and aggregating them, this module
    directly passes the molecular representation to the classifier.
    """

    def __init__(self):
        super().__init__()

    def forward(self, graph_repr, substructure_repr):
        num_graphs = graph_repr.size(0)
        attention_weights = torch.ones(num_graphs, device=graph_repr.device) / max(num_graphs, 1)

        return {
            "enhanced_graph_repr": graph_repr,
            "molecule_attention_weights": attention_weights,
            "top_substructure_indices": torch.arange(num_graphs, device=graph_repr.device),
            "top_substructure_weights": attention_weights,
        }


def get_args():
    parser = argparse.ArgumentParser(description="Ablation analysis for HMAN")

    # Dataset
    parser.add_argument("--dataset", type=str, default="tox21",
                        choices=["tox21", "sider", "muv", "toxcast"])
    parser.add_argument("--test_dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--run_task", type=int, default=-1)

    # Few-shot
    parser.add_argument("--n_shot_train", type=int, default=10)
    parser.add_argument("--n_shot_test", type=int, default=10)
    parser.add_argument("--n_query", type=int, default=16)
    parser.add_argument("--n_way", type=int, default=2)

    # Training
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--meta_batch_size", type=int, default=9)
    parser.add_argument("--inner_lr", type=float, default=0.1)
    parser.add_argument("--outer_lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--inner_steps", type=int, default=1)
    parser.add_argument("--outer_steps", type=int, default=1)
    parser.add_argument("--test_steps", type=int, default=1)

    # Model
    parser.add_argument("--gnn_type", type=str, default="gin",
                        choices=["gin", "gcn", "gat", "graphsage"])
    parser.add_argument("--num_gnn_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--attention_dim", type=int, default=300)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--top_b_atoms", type=int, default=5)
    parser.add_argument("--top_k_substructures", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num_substructure_samples", type=int, default=5)

    # Pretrained GNN
    parser.add_argument("--use_pretrained_gnn", action="store_true", default=True)
    parser.add_argument("--pretrained_gnn_path", type=str,
                        default="./chem_lib/model_gin/supervised_contextpred.pth")

    # System
    parser.add_argument("--result_dir", type=str, default="./results_ablation")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--save_model", action="store_true")

    return parser.parse_args()


def build_model(args, variant: str) -> HMAN:
    model = HMAN(
        gnn_type=args.gnn_type,
        num_gnn_layers=args.num_gnn_layers,
        emb_dim=args.emb_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        pooling=args.pooling,
        top_b_atoms=args.top_b_atoms,
        top_k_substructures=args.top_k_substructures,
        use_pretrained_gnn=args.use_pretrained_gnn,
        pretrained_gnn_path=args.pretrained_gnn_path,
    )

    if variant == "no_atom_attention":
        model.atom_attention = UniformAtomLevelAttention(top_b_atoms=args.top_b_atoms)

    if variant == "no_molecule_attention":
        model.molecule_attention = IdentityMoleculeLevelAttention()

    return model


def run_one_variant(args, variant: str) -> Dict[str, float]:
    set_seed(args.seed)

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    exp_dir = create_experiment_dir(args.result_dir, f"{args.dataset}_{variant}")
    logger = set_logger(exp_dir)
    save_config(args, exp_dir)

    logger.info("=" * 80)
    logger.info(f"Running ablation variant: {variant}")
    logger.info("=" * 80)

    train_tasks, valid_tasks, test_tasks = build_few_shot_tasks(
        dataset=args.dataset,
        data_dir=args.data_dir,
        test_dataset=args.test_dataset,
        run_task=args.run_task,
        seed=args.seed,
    )

    beta = 0.0 if variant == "no_weighted_nll_loss" else args.beta

    model = build_model(args, variant).to(device)
    count_model_params(model)

    trainer = HMANTrainer(
        model=model,
        train_tasks=train_tasks,
        valid_tasks=valid_tasks,
        test_tasks=test_tasks,
        device=device,
        n_way=args.n_way,
        n_shot_train=args.n_shot_train,
        n_shot_test=args.n_shot_test,
        n_query=args.n_query,
        meta_batch_size=args.meta_batch_size,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        weight_decay=args.weight_decay,
        inner_steps=args.inner_steps,
        outer_steps=args.outer_steps,
        test_steps=args.test_steps,
        beta=beta,
        num_substructure_samples=args.num_substructure_samples,
        second_order=False,
        exp_dir=exp_dir,
        logger=logger,
    )

    best_valid_auc = trainer.evaluate(split="valid")
    best_test_auc = trainer.evaluate(split="test")

    logger.info(f"Initial valid AUC: {best_valid_auc:.4f}")
    logger.info(f"Initial test AUC: {best_test_auc:.4f}")

    for epoch in range(1, args.epochs + 1):
        train_info = trainer.train_epoch(epoch)

        logger.info(
            f"Epoch [{epoch:04d}/{args.epochs}] "
            f"loss={train_info['loss']:.6f} "
            f"class_loss={train_info.get('class_loss', 0.0):.6f} "
            f"weighted_nll_loss={train_info.get('weighted_nll_loss', 0.0):.6f}"
        )

        if epoch == 1 or epoch % args.eval_steps == 0 or epoch == args.epochs:
            valid_auc = trainer.evaluate(split="valid")
            test_auc = trainer.evaluate(split="test")

            logger.info(
                f"Evaluation at epoch {epoch}: "
                f"valid_auc={valid_auc:.4f}, test_auc={test_auc:.4f}"
            )

            if valid_auc >= best_valid_auc:
                best_valid_auc = valid_auc
                best_test_auc = test_auc

                if args.save_model:
                    trainer.save_model("best_model.pt")

    trainer.save_result_log()

    result = {
        "variant": variant,
        "best_valid_auc": float(best_valid_auc),
        "corresponding_test_auc": float(best_test_auc),
    }

    logger.info(f"Final result: {result}")

    return result


def main():
    args = get_args()

    variants = [
        "full_hman",
        "no_atom_attention",
        "no_molecule_attention",
        "no_weighted_nll_loss",
    ]

    all_results = []

    for variant in variants:
        result = run_one_variant(args, variant)
        all_results.append(result)

    os.makedirs(args.result_dir, exist_ok=True)
    save_path = os.path.join(args.result_dir, f"ablation_summary_{args.dataset}.txt")

    with open(save_path, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(
                f"{result['variant']}\t"
                f"best_valid_auc={result['best_valid_auc']:.4f}\t"
                f"test_auc={result['corresponding_test_auc']:.4f}\n"
            )

    print("=" * 80)
    print("Ablation summary")
    print("=" * 80)
    for result in all_results:
        print(result)
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
