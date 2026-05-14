# model.py
import os
import copy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import (
        MessagePassing,
        global_add_pool,
        global_mean_pool,
        global_max_pool,
    )
    from torch_geometric.utils import add_self_loops, softmax
    from torch_scatter import scatter_add
except ImportError as e:
    raise ImportError(
        "This model requires torch_geometric and torch_scatter."
    ) from e

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None


# ============================================================
# Basic molecular GNN encoder
# Adapted from the original encoder.py
# ============================================================

NUM_ATOM_TYPE = 120
NUM_CHIRALITY_TAG = 3

NUM_BOND_TYPE = 6
NUM_BOND_DIRECTION = 3


class GINConv(MessagePassing):
    def __init__(self, emb_dim: int, aggr: str = "add"):
        super().__init__(aggr=aggr)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )

        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device)
        self_loop_attr[:, 0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.dtype)

        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        edge_embeddings = (
            self.edge_embedding1(edge_attr[:, 0])
            + self.edge_embedding2(edge_attr[:, 1])
        )

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):
    def __init__(self, emb_dim: int, aggr: str = "add"):
        super().__init__(aggr=aggr)

        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def norm(self, edge_index, num_nodes, dtype):
        edge_weight = torch.ones(
            edge_index.size(1),
            dtype=dtype,
            device=edge_index.device,
        )

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device)
        self_loop_attr[:, 0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.dtype)

        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        edge_embeddings = (
            self.edge_embedding1(edge_attr[:, 0])
            + self.edge_embedding2(edge_attr[:, 1])
        )

        norm = self.norm(edge_index, x.size(0), x.dtype)
        x = self.linear(x)

        return self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_embeddings,
            norm=norm,
        )

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(
        self,
        emb_dim: int,
        heads: int = 2,
        negative_slope: float = 0.2,
        aggr: str = "add",
    ):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(emb_dim, heads * emb_dim)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))
        self.bias = nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, heads * emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, heads * emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device)
        self_loop_attr[:, 0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.dtype)

        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        edge_embeddings = (
            self.edge_embedding1(edge_attr[:, 0])
            + self.edge_embedding2(edge_attr[:, 1])
        )

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j = x_j + edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        return aggr_out.mean(dim=1) + self.bias


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim: int, aggr: str = "mean"):
        super().__init__(aggr=aggr)

        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device)
        self_loop_attr[:, 0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.dtype)

        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        edge_embeddings = (
            self.edge_embedding1(edge_attr[:, 0])
            + self.edge_embedding2(edge_attr[:, 1])
        )

        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class MolecularGNN(nn.Module):
    def __init__(
        self,
        num_layer: int,
        emb_dim: int,
        JK: str = "last",
        drop_ratio: float = 0.0,
        gnn_type: str = "gin",
        batch_norm: bool = True,
    ):
        super().__init__()

        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.JK = JK
        self.drop_ratio = drop_ratio
        self.batch_norm = batch_norm

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList()

        for _ in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))
            else:
                raise ValueError(f"Unsupported gnn_type: {gnn_type}")

        if batch_norm:
            self.batch_norms = nn.ModuleList(
                [nn.BatchNorm1d(emb_dim) for _ in range(num_layer)]
            )

    def forward(self, x, edge_index, edge_attr):
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [h]

        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)

            if self.batch_norm:
                h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            h_list.append(h)

        if self.JK == "last":
            return h_list[-1]
        if self.JK == "sum":
            return torch.stack(h_list, dim=0).sum(dim=0)
        if self.JK == "max":
            return torch.stack(h_list, dim=0).max(dim=0)[0]
        if self.JK == "concat":
            return torch.cat(h_list, dim=1)

        raise ValueError(f"Unsupported JK mode: {self.JK}")


class MolecularEncoder(nn.Module):
    def __init__(
        self,
        num_layer: int,
        emb_dim: int,
        JK: str = "last",
        drop_ratio: float = 0.0,
        graph_pooling: str = "mean",
        gnn_type: str = "gin",
        batch_norm: bool = True,
    ):
        super().__init__()

        self.gnn = MolecularGNN(
            num_layer=num_layer,
            emb_dim=emb_dim,
            JK=JK,
            drop_ratio=drop_ratio,
            gnn_type=gnn_type,
            batch_norm=batch_norm,
        )

        if JK == "concat":
            self.out_dim = (num_layer + 1) * emb_dim
        else:
            self.out_dim = emb_dim

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unsupported graph_pooling: {graph_pooling}")

    def from_pretrained(self, model_file: str, device: str = "cpu"):
        if not os.path.exists(model_file):
            print(f"[Warning] pretrained GNN file not found: {model_file}")
            return

        state_dict = torch.load(model_file, map_location=device)

        try:
            self.gnn.load_state_dict(state_dict)
            print(f"Loaded pretrained GNN from {model_file}")
        except RuntimeError as e:
            print(f"[Warning] Failed to load pretrained GNN strictly: {e}")
            self.gnn.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained GNN with strict=False from {model_file}")

    def forward(self, data):
        node_repr = self.gnn(data.x, data.edge_index, data.edge_attr)
        graph_repr = self.pool(node_repr, data.batch)

        return graph_repr, node_repr


# ============================================================
# HMAN modules
# ============================================================

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        if num_layers <= 1:
            self.net = nn.Linear(in_dim, out_dim)
            return

        layers = []
        dim = in_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        layers.append(nn.Linear(dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AtomLevelAttention(nn.Module):
    """
    Atom-level attention in HMAN.

    It uses atom features, molecular features, and task prototype features
    to compute task-specific atom attention scores.
    """

    def __init__(
        self,
        emb_dim: int,
        attention_dim: int,
        top_b_atoms: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.top_b_atoms = top_b_atoms

        self.state_mlp = MLP(
            in_dim=emb_dim * 3,
            hidden_dim=attention_dim,
            out_dim=attention_dim,
            num_layers=2,
            dropout=dropout,
        )

        self.score_layer = nn.Linear(attention_dim, 1)

    def forward(
        self,
        node_repr: torch.Tensor,
        graph_repr: torch.Tensor,
        prototypes: torch.Tensor,
        batch: torch.Tensor,
    ):
        """
        Args:
            node_repr: [num_nodes, emb_dim]
            graph_repr: [num_graphs, emb_dim]
            prototypes: [2, emb_dim]
            batch: [num_nodes], graph id for each atom

        Returns:
            substructure_repr: [num_graphs, emb_dim]
            atom_weight_per_graph: list of tensors
            selected_atom_indices: list of tensors
            selected_atom_weights: list of tensors
        """
        device = node_repr.device
        num_graphs = graph_repr.size(0)

        substructure_repr_list = []
        atom_weight_per_graph = []
        selected_atom_indices = []
        selected_atom_weights = []

        proto_context = prototypes.mean(dim=0)

        for graph_id in range(num_graphs):
            atom_mask = batch == graph_id
            atom_indices = torch.nonzero(atom_mask, as_tuple=False).view(-1)

            atoms = node_repr[atom_indices]
            n_atoms = atoms.size(0)

            mol_feat = graph_repr[graph_id].unsqueeze(0).repeat(n_atoms, 1)
            proto_feat = proto_context.unsqueeze(0).repeat(n_atoms, 1)

            state_feat = torch.cat([atoms, mol_feat, proto_feat], dim=-1)

            hidden = self.state_mlp(state_feat)
            score = self.score_layer(hidden).view(-1)

            atom_weights = F.softmax(score, dim=0)

            k = min(self.top_b_atoms, n_atoms)
            top_weights, top_local_indices = torch.topk(atom_weights, k=k)

            top_global_indices = atom_indices[top_local_indices]
            selected_atoms = atoms[top_local_indices]

            sub_repr = torch.sum(
                selected_atoms * top_weights.unsqueeze(-1),
                dim=0,
            )

            substructure_repr_list.append(sub_repr)
            atom_weight_per_graph.append(atom_weights)
            selected_atom_indices.append(top_global_indices)
            selected_atom_weights.append(top_weights)

        substructure_repr = torch.stack(substructure_repr_list, dim=0)

        return {
            "substructure_repr": substructure_repr,
            "atom_weight_per_graph": atom_weight_per_graph,
            "selected_atom_indices": selected_atom_indices,
            "selected_atom_weights": selected_atom_weights,
        }


class MoleculeLevelAttention(nn.Module):
    """
    Molecule-level attention in HMAN.

    It uses selected substructure features as query and molecular features
    as key/value to produce property-aware molecular representations.
    """

    def __init__(
        self,
        emb_dim: int,
        attention_dim: int,
        top_k_substructures: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.top_k_substructures = top_k_substructures

        self.query_proj = nn.Linear(emb_dim, attention_dim)
        self.key_proj = nn.Linear(emb_dim, attention_dim)
        self.value_proj = nn.Linear(emb_dim, emb_dim)

        self.pattern_mlp = MLP(
            in_dim=emb_dim,
            hidden_dim=attention_dim,
            out_dim=emb_dim,
            num_layers=2,
            dropout=dropout,
        )

        self.fusion_mlp = MLP(
            in_dim=emb_dim * 2,
            hidden_dim=attention_dim,
            out_dim=emb_dim,
            num_layers=2,
            dropout=dropout,
        )

    def forward(
        self,
        graph_repr: torch.Tensor,
        substructure_repr: torch.Tensor,
    ):
        """
        Args:
            graph_repr: [num_graphs, emb_dim]
            substructure_repr: [num_graphs, emb_dim]

        Returns:
            enhanced_graph_repr: [num_graphs, emb_dim]
            molecule_attention_weights: [num_graphs]
        """
        q = self.query_proj(substructure_repr)
        k = self.key_proj(graph_repr)
        v = self.value_proj(substructure_repr)

        score = (q * k).sum(dim=-1) / (q.size(-1) ** 0.5)
        attention_weights = F.softmax(score, dim=0)

        k_top = min(self.top_k_substructures, graph_repr.size(0))
        top_weights, top_indices = torch.topk(attention_weights, k=k_top)

        pattern_context = torch.sum(
            v[top_indices] * top_weights.unsqueeze(-1),
            dim=0,
        )

        pattern_context = self.pattern_mlp(pattern_context)

        pattern_context = pattern_context.unsqueeze(0).repeat(graph_repr.size(0), 1)

        enhanced_graph_repr = self.fusion_mlp(
            torch.cat([graph_repr, pattern_context], dim=-1)
        )

        return {
            "enhanced_graph_repr": enhanced_graph_repr,
            "molecule_attention_weights": attention_weights,
            "top_substructure_indices": top_indices,
            "top_substructure_weights": top_weights,
        }


class HMAN(nn.Module):
    """
    Hierarchical Molecular Attention Network.

    The model follows the paper logic:
        1. GNN extracts atomic and molecular features.
        2. Prototypes are computed from support molecular features.
        3. Atom-level attention selects top-B atoms.
        4. Molecule-level attention selects top-k substructures.
        5. The enhanced molecular features are used for binary prediction.
    """

    def __init__(
        self,
        gnn_type: str = "gin",
        num_gnn_layers: int = 5,
        emb_dim: int = 300,
        attention_dim: int = 300,
        dropout: float = 0.5,
        pooling: str = "mean",
        top_b_atoms: int = 5,
        top_k_substructures: int = 3,
        use_pretrained_gnn: bool = True,
        pretrained_gnn_path: Optional[str] = None,
        JK: str = "last",
        batch_norm: bool = True,
        num_classes: int = 2,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.top_b_atoms = top_b_atoms
        self.top_k_substructures = top_k_substructures

        self.mol_encoder = MolecularEncoder(
            num_layer=num_gnn_layers,
            emb_dim=emb_dim,
            JK=JK,
            drop_ratio=dropout,
            graph_pooling=pooling,
            gnn_type=gnn_type,
            batch_norm=batch_norm,
        )

        encoder_out_dim = self.mol_encoder.out_dim

        if encoder_out_dim != emb_dim:
            self.encoder_proj = nn.Linear(encoder_out_dim, emb_dim)
        else:
            self.encoder_proj = nn.Identity()

        if use_pretrained_gnn and pretrained_gnn_path is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.mol_encoder.from_pretrained(pretrained_gnn_path, device=device)

        self.atom_attention = AtomLevelAttention(
            emb_dim=emb_dim,
            attention_dim=attention_dim,
            top_b_atoms=top_b_atoms,
            dropout=dropout,
        )

        self.molecule_attention = MoleculeLevelAttention(
            emb_dim=emb_dim,
            attention_dim=attention_dim,
            top_k_substructures=top_k_substructures,
            dropout=dropout,
        )

        self.classifier = nn.Linear(emb_dim, num_classes)

    def encode(self, data):
        graph_repr, node_repr = self.mol_encoder(data)

        graph_repr = self.encoder_proj(graph_repr)
        node_repr = self.encoder_proj(node_repr)

        return graph_repr, node_repr

    @staticmethod
    def compute_prototypes(
        support_graph_repr: torch.Tensor,
        support_labels: torch.Tensor,
        num_classes: int = 2,
    ):
        prototypes = []

        for c in range(num_classes):
            mask = support_labels == c

            if mask.sum() == 0:
                proto = support_graph_repr.mean(dim=0)
            else:
                proto = support_graph_repr[mask].mean(dim=0)

            prototypes.append(proto)

        return torch.stack(prototypes, dim=0)

    def forward(
        self,
        support_data,
        support_labels: torch.Tensor,
        query_data,
    ):
        support_graph_repr, support_node_repr = self.encode(support_data)
        query_graph_repr, query_node_repr = self.encode(query_data)

        prototypes = self.compute_prototypes(
            support_graph_repr,
            support_labels,
            num_classes=self.num_classes,
        )

        support_atom_out = self.atom_attention(
            node_repr=support_node_repr,
            graph_repr=support_graph_repr,
            prototypes=prototypes,
            batch=support_data.batch,
        )

        query_atom_out = self.atom_attention(
            node_repr=query_node_repr,
            graph_repr=query_graph_repr,
            prototypes=prototypes,
            batch=query_data.batch,
        )

        support_mol_out = self.molecule_attention(
            graph_repr=support_graph_repr,
            substructure_repr=support_atom_out["substructure_repr"],
        )

        query_mol_out = self.molecule_attention(
            graph_repr=query_graph_repr,
            substructure_repr=query_atom_out["substructure_repr"],
        )

        support_logits = self.classifier(support_mol_out["enhanced_graph_repr"])
        query_logits = self.classifier(query_mol_out["enhanced_graph_repr"])

        return {
            "support_logits": support_logits,
            "query_logits": query_logits,
            "support_graph_repr": support_graph_repr,
            "query_graph_repr": query_graph_repr,
            "prototypes": prototypes,
            "support_atom_out": support_atom_out,
            "query_atom_out": query_atom_out,
            "support_molecule_out": support_mol_out,
            "query_molecule_out": query_mol_out,
        }

    def get_adaptable_parameters(self):
        """
        According to the paper, inner optimization mainly updates
        atom-level attention parameters Phi.
        """
        return list(self.atom_attention.parameters())


# ============================================================
# Losses
# ============================================================

def classification_loss(logits: torch.Tensor, labels: torch.Tensor):
    return F.cross_entropy(logits, labels.long())


def weighted_negative_log_likelihood_loss(
    atom_out: Dict,
    logits: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-8,
):
    """
    Weighted negative log-likelihood loss.

    The paper defines this loss to align attention-derived weight distribution
    with reward/predicted probability distribution.

    Here:
        W(xi): selected atom attention weights of each molecule.
        R(xi): predicted probability of the ground-truth class.

    A practical differentiable approximation is:
        L = - mean_i [ R_i * log(W_i) ]

    where W_i is the mean selected attention weight of the selected top-B atoms
    for molecule i.
    """
    probs = F.softmax(logits, dim=-1)
    reward = probs.gather(1, labels.long().view(-1, 1)).view(-1)
    reward = reward.detach()

    selected_weights = atom_out["selected_atom_weights"]

    graph_weight_list = []

    for weights in selected_weights:
        graph_weight = weights.mean()
        graph_weight_list.append(graph_weight)

    weight_score = torch.stack(graph_weight_list, dim=0)
    weight_score = torch.clamp(weight_score, min=eps, max=1.0)

    loss = -torch.mean(reward * torch.log(weight_score))

    return loss


# ============================================================
# Trainer
# ============================================================

class HMANTrainer:
    def __init__(
        self,
        model: HMAN,
        train_tasks,
        valid_tasks,
        test_tasks,
        device: str,
        n_way: int = 2,
        n_shot_train: int = 10,
        n_shot_test: int = 10,
        n_query: int = 16,
        meta_batch_size: int = 9,
        inner_lr: float = 0.1,
        outer_lr: float = 0.001,
        weight_decay: float = 5e-5,
        inner_steps: int = 1,
        outer_steps: int = 1,
        test_steps: int = 1,
        beta: float = 0.1,
        num_substructure_samples: int = 5,
        second_order: bool = False,
        exp_dir: Optional[str] = None,
        logger=None,
    ):
        self.model = model
        self.train_tasks = train_tasks
        self.valid_tasks = valid_tasks
        self.test_tasks = test_tasks

        self.device = device
        self.n_way = n_way
        self.n_shot_train = n_shot_train
        self.n_shot_test = n_shot_test
        self.n_query = n_query
        self.meta_batch_size = meta_batch_size

        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.weight_decay = weight_decay

        self.inner_steps = inner_steps
        self.outer_steps = outer_steps
        self.test_steps = test_steps

        self.beta = beta
        self.num_substructure_samples = num_substructure_samples
        self.second_order = second_order

        self.exp_dir = exp_dir
        self.logger = logger

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.outer_lr,
            weight_decay=self.weight_decay,
        )

        self.history = {
            "train_loss": [],
            "valid_auc": [],
            "test_auc": [],
        }

    def _log(self, message: str):
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def _support_objective(self, output, support_labels):
        c_loss = classification_loss(
            output["support_logits"],
            support_labels,
        )

        nll_loss = weighted_negative_log_likelihood_loss(
            atom_out=output["support_atom_out"],
            logits=output["support_logits"],
            labels=support_labels,
        )

        total_loss = c_loss + self.beta * nll_loss

        return total_loss, c_loss, nll_loss

    def _query_objective(self, output, query_labels):
        return classification_loss(
            output["query_logits"],
            query_labels,
        )

    def _adapt_model(self, model, episode, train_mode: bool = True):
        """
        Lightweight inner adaptation.

        To keep the code self-contained and avoid learn2learn dependency,
        this function performs a first-order style temporary update by
        optimizing a copied model during inner-loop adaptation.

        This is simpler and safer for code cleanup, but if you want exact
        second-order MAML, we can later replace this part with your original
        maml.py logic.
        """
        adapted_model = copy.deepcopy(model)
        adapted_model.to(self.device)
        adapted_model.train()

        inner_optimizer = torch.optim.SGD(
            adapted_model.get_adaptable_parameters(),
            lr=self.inner_lr,
        )

        last_c_loss = None
        last_nll_loss = None
        last_total_loss = None

        for _ in range(self.inner_steps if train_mode else self.test_steps):
            output = adapted_model(
                episode.support_data,
                episode.support_labels,
                episode.query_data,
            )

            total_loss, c_loss, nll_loss = self._support_objective(
                output,
                episode.support_labels,
            )

            inner_optimizer.zero_grad()
            total_loss.backward()
            inner_optimizer.step()

            last_total_loss = total_loss.detach()
            last_c_loss = c_loss.detach()
            last_nll_loss = nll_loss.detach()

        return adapted_model, {
            "support_loss": float(last_total_loss.item()) if last_total_loss is not None else 0.0,
            "support_class_loss": float(last_c_loss.item()) if last_c_loss is not None else 0.0,
            "support_weighted_nll_loss": float(last_nll_loss.item()) if last_nll_loss is not None else 0.0,
        }

    def train_epoch(self, epoch: int):
        self.model.train()

        if len(self.train_tasks) == 0:
            raise ValueError("No training tasks are available.")

        task_batch = random.sample(
            self.train_tasks,
            k=min(self.meta_batch_size, len(self.train_tasks)),
        )

        total_loss = 0.0
        total_class_loss = 0.0
        total_weighted_nll_loss = 0.0

        self.optimizer.zero_grad()

        for task in task_batch:
            episode = task.sample_train_episode(
                n_shot=self.n_shot_train,
                n_query=self.n_query,
                device=self.device,
            )

            # Inner adaptation on a copied model.
            adapted_model, adapt_info = self._adapt_model(
                self.model,
                episode,
                train_mode=True,
            )

            adapted_model.eval()

            output = adapted_model(
                episode.support_data,
                episode.support_labels,
                episode.query_data,
            )

            query_loss = self._query_objective(
                output,
                episode.query_labels,
            )

            # Because adapted_model is a deep copy, query_loss does not
            # backpropagate to the original model through the inner step.
            # We therefore also compute a direct query loss on the original
            # model to update outer parameters.
            original_output = self.model(
                episode.support_data,
                episode.support_labels,
                episode.query_data,
            )

            outer_query_loss = self._query_objective(
                original_output,
                episode.query_labels,
            )

            support_total, support_c, support_nll = self._support_objective(
                original_output,
                episode.support_labels,
            )

            loss = outer_query_loss + 0.1 * support_total
            loss = loss / len(task_batch)

            loss.backward()

            total_loss += float(loss.detach().item())
            total_class_loss += float(support_c.detach().item())
            total_weighted_nll_loss += float(support_nll.detach().item())

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        avg_class_loss = total_class_loss / len(task_batch)
        avg_weighted_nll_loss = total_weighted_nll_loss / len(task_batch)

        self.history["train_loss"].append(total_loss)

        return {
            "loss": total_loss,
            "class_loss": avg_class_loss,
            "weighted_nll_loss": avg_weighted_nll_loss,
        }

    @torch.no_grad()
    def _predict_episode(self, model, episode):
        model.eval()

        output = model(
            episode.support_data,
            episode.support_labels,
            episode.query_data,
        )

        probs = F.softmax(output["query_logits"], dim=-1)[:, 1]

        return probs.detach().cpu(), episode.query_labels.detach().cpu()

    def evaluate(self, split: str = "test"):
        if split == "valid":
            tasks = self.valid_tasks
            n_shot = self.n_shot_test
        elif split == "test":
            tasks = self.test_tasks
            n_shot = self.n_shot_test
        elif split == "train":
            tasks = self.train_tasks
            n_shot = self.n_shot_train
        else:
            raise ValueError(f"Unsupported split: {split}")

        if len(tasks) == 0:
            return 0.0

        auc_scores = []

        for task in tasks:
            episode = task.sample_test_episode(
                n_shot=n_shot,
                n_query=None,
                device=self.device,
            )

            adapted_model, _ = self._adapt_model(
                self.model,
                episode,
                train_mode=False,
            )

            y_score, y_true = self._predict_episode(adapted_model, episode)

            auc = self._safe_auc(y_true.numpy(), y_score.numpy())
            auc_scores.append(auc)

        avg_auc = float(np.mean(auc_scores))

        if split == "valid":
            self.history["valid_auc"].append(avg_auc)
        elif split == "test":
            self.history["test_auc"].append(avg_auc)

        return avg_auc

    @staticmethod
    def _safe_auc(y_true, y_score):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        if len(np.unique(y_true)) < 2:
            return 0.5

        if roc_auc_score is None:
            # Simple fallback: return 0.5 if sklearn is not available.
            return 0.5

        try:
            return float(roc_auc_score(y_true, y_score))
        except ValueError:
            return 0.5

    def save_model(self, filename: str = "model.pt"):
        if self.exp_dir is None:
            save_path = filename
        else:
            os.makedirs(self.exp_dir, exist_ok=True)
            save_path = os.path.join(self.exp_dir, filename)

        torch.save(self.model.state_dict(), save_path)
        self._log(f"Model saved to {save_path}")

    def load_model(self, filename: str):
        self.model.load_state_dict(
            torch.load(filename, map_location=self.device)
        )
        self.model.to(self.device)

    def conclude(self):
        self._log("HMAN training/evaluation completed.")

    def save_result_log(self):
        if self.exp_dir is None:
            save_path = "training_history.pt"
        else:
            os.makedirs(self.exp_dir, exist_ok=True)
            save_path = os.path.join(self.exp_dir, "training_history.pt")

        torch.save(self.history, save_path)
        self._log(f"Training history saved to {save_path}")