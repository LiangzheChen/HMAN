# visualize.py
"""
Visualization script for HMAN.

This script provides two visualization procedures:
    1. t-SNE visualization of molecular property embeddings.
    2. Atom-level attention visualization on molecular structures.

It assumes:
    main.py
    data.py
    model.py
    util.py

Example:
    python visualize.py \
        --dataset tox21 \
        --data_dir ./data/ \
        --checkpoint ./results/tox21/xxxx/best_model.pt \
        --task_id 9 \
        --gpu_id 0
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
except ImportError:
    Chem = None
    Draw = None

from data import build_few_shot_tasks, batch_graphs
from model import HMAN
from util import ensure_dir, set_seed


def get_args():
    parser = argparse.ArgumentParser(description="HMAN visualization")

    # Dataset
    parser.add_argument("--dataset", type=str, default="tox21",
                        choices=["tox21", "sider", "muv", "toxcast"])
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--task_id", type=int, default=-1,
                        help="Task id to visualize. If -1, use the first test task.")

    # Few-shot
    parser.add_argument("--n_shot", type=int, default=10)
    parser.add_argument("--n_query", type=int, default=200,
                        help="Number of query molecules used for t-SNE. Use -1 for all query molecules.")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gnn_type", type=str, default="gin",
                        choices=["gin", "gcn", "gat", "graphsage"])
    parser.add_argument("--num_gnn_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--attention_dim", type=int, default=300)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--top_b_atoms", type=int, default=5)
    parser.add_argument("--top_k_substructures", type=int, default=3)
    parser.add_argument("--use_pretrained_gnn", action="store_true", default=True)
    parser.add_argument("--pretrained_gnn_path", type=str,
                        default="./chem_lib/model_gin/supervised_contextpred.pth")

    # Output
    parser.add_argument("--output_dir", type=str, default="./visualization_results")
    parser.add_argument("--num_molecule_vis", type=int, default=8,
                        help="Number of molecules for atom attention visualization.")

    # System
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0)

    return parser.parse_args()


def load_model(args, device):
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
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def choose_task(args, test_tasks):
    if len(test_tasks) == 0:
        raise ValueError("No test tasks are available.")

    if args.task_id < 0:
        return test_tasks[0]

    for task in test_tasks:
        if task.task_id == args.task_id:
            return task

    raise ValueError(f"Task id {args.task_id} not found in test tasks.")


@torch.no_grad()
def extract_query_embeddings(model, episode):
    output = model(
        episode.support_data,
        episode.support_labels,
        episode.query_data,
    )

    embeddings = output["query_molecule_out"]["enhanced_graph_repr"]
    logits = output["query_logits"]
    probs = F.softmax(logits, dim=-1)[:, 1]

    labels = episode.query_labels

    return embeddings.detach().cpu().numpy(), labels.detach().cpu().numpy(), probs.detach().cpu().numpy(), output


def plot_tsne(embeddings, labels, probs, save_path):
    if embeddings.shape[0] < 3:
        raise ValueError("At least 3 samples are required for t-SNE visualization.")

    perplexity = min(30, max(2, embeddings.shape[0] // 5))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=5,
    )

    z = tsne.fit_transform(embeddings)

    plt.figure(figsize=(7, 6))

    neg_mask = labels == 0
    pos_mask = labels == 1

    plt.scatter(z[neg_mask, 0], z[neg_mask, 1], label="Inactive / Negative", alpha=0.75)
    plt.scatter(z[pos_mask, 0], z[pos_mask, 1], label="Active / Positive", alpha=0.75)

    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title("HMAN molecular property feature visualization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def get_query_smiles_from_task(task, episode_query_labels, max_num: int):
    """
    Recover several query molecules for RDKit visualization.

    Because PyG Batch may not always preserve string attributes safely,
    this function directly samples molecules from the task dataset.
    """
    selected = []

    for idx in range(len(task.dataset)):
        data = task.dataset[idx]
        if not hasattr(data, "smiles"):
            continue

        selected.append((idx, data.smiles, int(data.y.view(-1)[0].item())))

        if len(selected) >= max_num:
            break

    return selected


@torch.no_grad()
def compute_single_molecule_attention(model, task, support_episode, molecule_index, device):
    """
    Compute atom-level attention for one molecule.

    We reuse the support set of the episode to compute prototypes,
    then feed a single query molecule to obtain atom attention.
    """
    data = task.dataset[molecule_index]
    query_batch = batch_graphs([data], device=device)

    output = model(
        support_episode.support_data,
        support_episode.support_labels,
        query_batch,
    )

    atom_out = output["query_atom_out"]

    if len(atom_out["selected_atom_indices"]) == 0:
        return []

    selected_global = atom_out["selected_atom_indices"][0].detach().cpu().numpy().tolist()
    selected_weights = atom_out["selected_atom_weights"][0].detach().cpu().numpy().tolist()

    # For a single graph batch, global atom indices are the same as local atom indices.
    return list(zip(selected_global, selected_weights))


def draw_attention_molecule(smiles: str, atom_weights, save_path: str):
    if Chem is None or Draw is None:
        raise ImportError("RDKit is required for molecular visualization.")

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return

    highlight_atoms = [int(idx) for idx, _ in atom_weights]
    highlight_bonds = []

    atom_set = set(highlight_atoms)

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()

        if begin in atom_set and end in atom_set:
            highlight_bonds.append(bond.GetIdx())

    img = Draw.MolToImage(
        mol,
        size=(500, 400),
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
    )

    img.save(save_path)


def save_attention_text(smiles: str, atom_weights, save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"SMILES: {smiles}\n")
        f.write("Selected atoms and attention weights:\n")
        for atom_idx, weight in atom_weights:
            f.write(f"atom_index={atom_idx}, weight={weight:.6f}\n")


def main():
    args = get_args()
    set_seed(args.seed)

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    output_dir = ensure_dir(args.output_dir)

    _, _, test_tasks = build_few_shot_tasks(
        dataset=args.dataset,
        data_dir=args.data_dir,
        test_dataset=None,
        run_task=-1,
        seed=args.seed,
    )

    task = choose_task(args, test_tasks)

    model = load_model(args, device)

    n_query = None if args.n_query < 0 else args.n_query

    episode = task.sample_test_episode(
        n_shot=args.n_shot,
        n_query=n_query,
        device=device,
    )

    # ========================================================
    # 1. t-SNE visualization
    # ========================================================
    embeddings, labels, probs, output = extract_query_embeddings(model, episode)

    tsne_path = os.path.join(
        output_dir,
        f"tsne_{args.dataset}_task_{task.task_id}.png",
    )

    plot_tsne(embeddings, labels, probs, tsne_path)

    print(f"t-SNE visualization saved to: {tsne_path}")

    # ========================================================
    # 2. Atom-level attention visualization
    # ========================================================
    selected_molecules = get_query_smiles_from_task(
        task=task,
        episode_query_labels=episode.query_labels,
        max_num=args.num_molecule_vis,
    )

    mol_vis_dir = ensure_dir(os.path.join(output_dir, "molecule_attention"))

    for rank, (mol_idx, smiles, label) in enumerate(selected_molecules):
        atom_weights = compute_single_molecule_attention(
            model=model,
            task=task,
            support_episode=episode,
            molecule_index=mol_idx,
            device=device,
        )

        txt_path = os.path.join(
            mol_vis_dir,
            f"task_{task.task_id}_mol_{rank}_label_{label}_attention.txt",
        )
        save_attention_text(smiles, atom_weights, txt_path)

        if Chem is not None and Draw is not None:
            img_path = os.path.join(
                mol_vis_dir,
                f"task_{task.task_id}_mol_{rank}_label_{label}.png",
            )
            draw_attention_molecule(smiles, atom_weights, img_path)
            print(f"Molecule attention figure saved to: {img_path}")

        print(f"Attention weights saved to: {txt_path}")

    print("Visualization completed.")


if __name__ == "__main__":
    main()
