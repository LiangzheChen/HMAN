# data.py
import os
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, Batch

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    Chem = None
    AllChem = None


# ============================================================
# Molecular graph construction
# ============================================================

if Chem is not None:
    ALLOWABLE_FEATURES = {
        "possible_atomic_num_list": list(range(1, 119)),
        "possible_chirality_list": [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER,
        ],
        "possible_bonds": [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ],
        "possible_bond_dirs": [
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT,
        ],
    }
else:
    ALLOWABLE_FEATURES = None


def mol_to_graph_data_obj_simple(mol) -> Data:
    """
    Convert an RDKit molecule into a PyTorch Geometric Data object.

    Node features:
        x[:, 0]: atomic number index
        x[:, 1]: chirality tag index

    Edge features:
        edge_attr[:, 0]: bond type index
        edge_attr[:, 1]: bond direction index
    """
    if Chem is None:
        raise ImportError("RDKit is required to construct molecular graphs.")

    atom_features_list = []

    for atom in mol.GetAtoms():
        atom_feature = [
            ALLOWABLE_FEATURES["possible_atomic_num_list"].index(atom.GetAtomicNum()),
            ALLOWABLE_FEATURES["possible_chirality_list"].index(atom.GetChiralTag()),
        ]
        atom_features_list.append(atom_feature)

    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = [
                ALLOWABLE_FEATURES["possible_bonds"].index(bond.GetBondType()),
                ALLOWABLE_FEATURES["possible_bond_dirs"].index(bond.GetBondDir()),
            ]

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)

            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ============================================================
# Dataset loading
# ============================================================

def _load_binary_json_dataset(input_path: str):
    """
    Load binary molecular task JSON.

    Expected format:
        [
            ["negative_smiles_1", "negative_smiles_2", ...],
            ["positive_smiles_1", "positive_smiles_2", ...]
        ]
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"JSON file does not exist: {input_path}")

    if os.path.getsize(input_path) == 0:
        raise ValueError(f"JSON file is empty: {input_path}")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            binary_list = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON file: {input_path}\n"
            f"Please check whether this file is a valid JSON file.\n"
            f"Expected format: [[negative_smiles...], [positive_smiles...]]"
        ) from e

    if not isinstance(binary_list, list) or len(binary_list) != 2:
        raise ValueError(
            f"Invalid JSON format in {input_path}. "
            f"Expected a list with two elements: [negative_smiles, positive_smiles]."
        )

    smiles_list = []
    labels = []

    for class_id, smiles_group in enumerate(binary_list):
        if not isinstance(smiles_group, list):
            raise ValueError(
                f"Invalid class group in {input_path}. "
                f"Each class should be a list of SMILES strings."
            )

        for smi in smiles_group:
            smiles_list.append(smi)
            labels.append(class_id)

    rdkit_mol_objs_list = [AllChem.MolFromSmiles(smi) for smi in smiles_list]

    labels = np.array(labels, dtype=np.int64).reshape(-1, 1)

    return smiles_list, rdkit_mol_objs_list, labels


class MoleculeTaskDataset(InMemoryDataset):
    """
    A task-specific molecular dataset.

    Each task is treated as a binary classification dataset.
    The raw file should contain the negative and positive SMILES lists.
    """

    def __init__(
        self,
        root: str,
        dataset: str,
        task_id: int,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        empty: bool = False,
    ):
        self.dataset = dataset
        self.task_id = task_id

        super().__init__(root, transform, pre_transform, pre_filter)

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
            smiles_path = os.path.join(self.processed_dir, "smiles.csv")
            self.smiles_list = pd.read_csv(smiles_path, header=None).to_numpy()[:, 0]

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        raise NotImplementedError("Please prepare raw molecular task files manually.")

    def process(self):
        if len(self.raw_paths) == 0:
            raise FileNotFoundError(f"No raw file found in {self.raw_dir}")

        json_paths = [
            path for path in self.raw_paths
            if path.lower().endswith(".json")
        ]

        if len(json_paths) == 0:
            raise FileNotFoundError(
                f"No JSON file found in raw directory: {self.raw_dir}. "
                f"Current raw files are: {self.raw_paths}"
            )

        input_path = json_paths[0]

        smiles_list, rdkit_mol_objs, labels = _load_binary_json_dataset(input_path)

        data_list = []
        data_smiles_list = []

        for i, mol in enumerate(rdkit_mol_objs):
            if mol is None:
                continue

            data = mol_to_graph_data_obj_simple(mol)
            data.id = torch.tensor([i], dtype=torch.long)
            data.y = torch.tensor(labels[i], dtype=torch.long)
            data.task_id = torch.tensor([self.task_id], dtype=torch.long)

            data_list.append(data)
            data_smiles_list.append(smiles_list[i])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        os.makedirs(self.processed_dir, exist_ok=True)

        pd.Series(data_smiles_list).to_csv(
            os.path.join(self.processed_dir, "smiles.csv"),
            index=False,
            header=False,
        )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        data = super().get(idx)
        data.smiles = self.smiles_list[idx]
        return data


# ============================================================
# Task split settings
# ============================================================

def obtain_train_test_tasks(dataset: str) -> Tuple[List[int], List[int]]:
    """
    Return train/test task ids for each dataset.

    This follows the original baseline task split.
    """
    dataset = dataset.lower()

    tox21_train_tasks = list(range(9))
    tox21_test_tasks = list(range(9, 12))

    sider_train_tasks = list(range(21))
    sider_test_tasks = list(range(21, 27))

    muv_train_tasks = list(range(12))
    muv_test_tasks = list(range(12, 17))

    toxcast_drop_tasks = [
        343, 348, 349, 352, 354, 355, 356, 357, 358, 360, 361, 362,
        364, 367, 368, 369, 370, 371, 372, 373, 374, 376, 377, 378,
        379, 380, 381, 382, 383, 384, 385, 387, 388, 389, 390, 391,
        392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403,
        404, 406, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
        418, 419, 420, 421, 422, 423, 424, 426, 428, 429, 430, 431,
        432, 433, 434, 435, 436, 437, 438, 439, 440, 442, 443, 444,
        445, 446, 447, 449, 450, 451, 452, 453, 474, 475, 477, 480,
        481, 482, 483,
    ]

    toxcast_train_tasks = [
        x for x in list(range(450)) if x not in toxcast_drop_tasks
    ]
    toxcast_test_tasks = [
        x for x in list(range(450, 617)) if x not in toxcast_drop_tasks
    ]

    if dataset == "tox21":
        return tox21_train_tasks, tox21_test_tasks
    if dataset == "sider":
        return sider_train_tasks, sider_test_tasks
    if dataset == "muv":
        return muv_train_tasks, muv_test_tasks
    if dataset == "toxcast":
        return toxcast_train_tasks, toxcast_test_tasks

    raise ValueError(f"Unsupported dataset: {dataset}")


# Keep the original misspelled function name for compatibility.
def obatin_train_test_tasks(dataset: str):
    return obtain_train_test_tasks(dataset)


# ============================================================
# Few-shot sampling
# ============================================================

def sample_indices(indices: Sequence[int], size: int) -> List[int]:
    """
    Sample indices with replacement-like recursion when the class size is small.

    This follows the behavior of the original code:
    if the available number is smaller than the required size,
    repeatedly sample until enough indices are obtained.
    """
    indices = list(indices)

    if len(indices) == 0:
        raise ValueError("Cannot sample from an empty index list.")

    if len(indices) >= size:
        return random.sample(indices, size)

    return random.sample(indices, len(indices)) + sample_indices(indices, size - len(indices))


def split_pos_neg_indices(dataset: MoleculeTaskDataset) -> Tuple[List[int], List[int]]:
    """
    Split molecule indices into negative and positive classes according to data.y.
    """
    neg_indices = []
    pos_indices = []

    for i in range(len(dataset)):
        y = int(dataset[i].y.view(-1)[0].item())

        if y == 0:
            neg_indices.append(i)
        elif y == 1:
            pos_indices.append(i)

    if len(neg_indices) == 0 or len(pos_indices) == 0:
        raise ValueError(
            f"Task {dataset.task_id} has invalid class distribution: "
            f"negative={len(neg_indices)}, positive={len(pos_indices)}"
        )

    return neg_indices, pos_indices


def batch_graphs(data_list: List[Data], device: Optional[str] = None) -> Batch:
    """
    Convert a list of PyG Data objects into a batched graph.
    """
    batch = Batch.from_data_list(data_list)

    if device is not None:
        batch = batch.to(device)

    return batch


@dataclass
class FewShotEpisode:
    support_data: Batch
    support_labels: torch.Tensor
    query_data: Batch
    query_labels: torch.Tensor
    task_id: int
    dataset_name: str


class MolecularFewShotTask:
    """
    A binary few-shot molecular property prediction task.

    It provides support/query sampling for meta-training and meta-testing.
    """

    def __init__(
        self,
        dataset_name: str,
        task_id: int,
        dataset: MoleculeTaskDataset,
    ):
        self.dataset_name = dataset_name
        self.task_id = task_id
        self.dataset = dataset

        self.neg_indices, self.pos_indices = split_pos_neg_indices(dataset)

    def sample_train_episode(
        self,
        n_shot: int,
        n_query: int,
        device: Optional[str] = None,
    ) -> FewShotEpisode:
        """
        Sample one episode for meta-training.

        Support set:
            n_shot negative + n_shot positive

        Query set:
            n_query samples from the remaining molecules
        """
        neg_support = sample_indices(self.neg_indices, n_shot)
        pos_support = sample_indices(self.pos_indices, n_shot)

        support_indices = neg_support + pos_support

        remaining_indices = [
            i for i in range(len(self.dataset)) if i not in set(support_indices)
        ]

        query_indices = sample_indices(remaining_indices, n_query)

        return self._make_episode(support_indices, query_indices, device=device)

    def sample_test_episode(
        self,
        n_shot: int,
        n_query: Optional[int] = None,
        device: Optional[str] = None,
    ) -> FewShotEpisode:
        """
        Sample one episode for evaluation.

        Support set:
            n_shot negative + n_shot positive

        Query set:
            If n_query is None, use all remaining molecules.
            Otherwise, sample n_query molecules from the remaining molecules.
        """
        neg_support = sample_indices(self.neg_indices, n_shot)
        pos_support = sample_indices(self.pos_indices, n_shot)

        support_indices = neg_support + pos_support

        remaining_indices = [
            i for i in range(len(self.dataset)) if i not in set(support_indices)
        ]

        if n_query is None:
            query_indices = remaining_indices
        else:
            query_indices = sample_indices(remaining_indices, n_query)

        return self._make_episode(support_indices, query_indices, device=device)

    def _make_episode(
        self,
        support_indices: List[int],
        query_indices: List[int],
        device: Optional[str] = None,
    ) -> FewShotEpisode:
        support_list = [self.dataset[i] for i in support_indices]
        query_list = [self.dataset[i] for i in query_indices]

        support_labels = torch.tensor(
            [int(data.y.view(-1)[0].item()) for data in support_list],
            dtype=torch.long,
        )

        query_labels = torch.tensor(
            [int(data.y.view(-1)[0].item()) for data in query_list],
            dtype=torch.long,
        )

        if device is not None:
            support_labels = support_labels.to(device)
            query_labels = query_labels.to(device)

        support_batch = batch_graphs(support_list, device=device)
        query_batch = batch_graphs(query_list, device=device)

        return FewShotEpisode(
            support_data=support_batch,
            support_labels=support_labels,
            query_data=query_batch,
            query_labels=query_labels,
            task_id=self.task_id,
            dataset_name=self.dataset_name,
        )


# ============================================================
# Dataset path handling
# ============================================================

def _find_task_root(data_dir: str, dataset: str, task_id: int) -> str:
    """
    Find the root directory for a task.

    Compatible with the original baseline format:
        data/{dataset}/new/{task_id + 1}/raw/*.json

    Also supports:
        data/{dataset}/{task_id}/raw/*.json
        data/{dataset}/{task_id}.json
    """
    dataset = dataset.lower()

    candidates = [
        # Original baseline format
        os.path.join(data_dir, dataset, "new", str(task_id + 1)),
        os.path.join(data_dir, dataset, "new", str(task_id)),

        # Cleaned possible formats
        os.path.join(data_dir, dataset, str(task_id)),
        os.path.join(data_dir, dataset, f"task_{task_id}"),
        os.path.join(data_dir, dataset, f"{task_id}"),
    ]

    for path in candidates:
        raw_dir = os.path.join(path, "raw")

        if os.path.isdir(raw_dir):
            json_files = [
                f for f in os.listdir(raw_dir)
                if f.lower().endswith(".json")
            ]

            if len(json_files) > 0:
                return path

    # Flat json format
    flat_candidates = [
        os.path.join(data_dir, dataset, f"{task_id}.json"),
        os.path.join(data_dir, dataset, f"{task_id + 1}.json"),
        os.path.join(data_dir, dataset, f"task_{task_id}.json"),
        os.path.join(data_dir, dataset, f"task_{task_id + 1}.json"),
        os.path.join(data_dir, dataset, "new", f"{task_id + 1}.json"),
    ]

    for raw_file in flat_candidates:
        if os.path.isfile(raw_file):
            task_root = os.path.join(data_dir, dataset, "new", str(task_id + 1))
            raw_dir = os.path.join(task_root, "raw")
            os.makedirs(raw_dir, exist_ok=True)

            target_file = os.path.join(raw_dir, os.path.basename(raw_file))

            if not os.path.exists(target_file):
                import shutil
                shutil.copy(raw_file, target_file)

            return task_root

    raise FileNotFoundError(
        f"Cannot find JSON raw data for dataset={dataset}, task_id={task_id}. "
        f"Expected format like:\n"
        f"  {data_dir}/{dataset}/new/{task_id + 1}/raw/*.json\n"
        f"or\n"
        f"  {data_dir}/{dataset}/{task_id}.json"
    )


def load_molecular_task(
    dataset_name: str,
    task_id: int,
    data_dir: str,
) -> MolecularFewShotTask:
    task_root = _find_task_root(data_dir, dataset_name, task_id)

    pyg_dataset = MoleculeTaskDataset(
        root=task_root,
        dataset=dataset_name,
        task_id=task_id,
    )

    return MolecularFewShotTask(
        dataset_name=dataset_name,
        task_id=task_id,
        dataset=pyg_dataset,
    )


# ============================================================
# Public API used by main.py
# ============================================================

def build_few_shot_tasks(
    dataset: str,
    data_dir: str,
    test_dataset: Optional[str] = None,
    run_task: int = -1,
    seed: int = 5,
    valid_ratio: float = 0.1,
):
    """
    Build train/valid/test molecular few-shot tasks.

    Returns:
        train_tasks: List[MolecularFewShotTask]
        valid_tasks: List[MolecularFewShotTask]
        test_tasks:  List[MolecularFewShotTask]

    This function is designed to match the main.py interface:

        train_tasks, valid_tasks, test_tasks = build_few_shot_tasks(...)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = dataset.lower()

    train_task_ids, test_task_ids = obtain_train_test_tasks(dataset)

    if test_dataset is not None:
        test_dataset = test_dataset.lower()

        train_ids_2, test_ids_2 = obtain_train_test_tasks(test_dataset)

        # Same logic as the original parser:
        # if cross-dataset evaluation is used, all source dataset tasks are
        # used for training, and all target dataset tasks are used for testing.
        train_task_ids = sorted(list(set(train_task_ids + test_task_ids)))
        test_task_ids = sorted(list(set(train_ids_2 + test_ids_2)))
        test_dataset_name = test_dataset
    else:
        test_dataset_name = dataset

    if run_task >= 0:
        train_task_ids = [run_task]
        test_task_ids = [run_task]
        test_dataset_name = dataset

    random.shuffle(train_task_ids)

    num_valid = max(1, int(len(train_task_ids) * valid_ratio)) if len(train_task_ids) > 1 else 0

    valid_task_ids = train_task_ids[:num_valid]
    real_train_task_ids = train_task_ids[num_valid:]

    train_tasks = [
        load_molecular_task(dataset, task_id, data_dir)
        for task_id in real_train_task_ids
    ]

    valid_tasks = [
        load_molecular_task(dataset, task_id, data_dir)
        for task_id in valid_task_ids
    ]

    test_tasks = [
        load_molecular_task(test_dataset_name, task_id, data_dir)
        for task_id in test_task_ids
    ]

    return train_tasks, valid_tasks, test_tasks