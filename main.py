# main.py
import os
import time
import argparse
import random
import numpy as np
import torch

from data import build_few_shot_tasks
from model import HMAN, HMANTrainer
from util import (
    count_model_params,
    create_experiment_dir,
    save_config,
    set_logger,
)


def set_seed(seed: int):
    """Set random seed for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(
        description="HMAN: Hierarchical Molecular Attention Network for Few-Shot Molecular Property Prediction"
    )

    # =========================
    # Dataset settings
    # =========================
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument(
        "--dataset",
        type=str,
        default="tox21",
        choices=["tox21", "sider", "muv", "toxcast", "bbbp", "flavormole"],
        help="Dataset used for few-shot molecular property prediction.",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default=None,
        help="Optional cross-dataset testing dataset.",
    )
    parser.add_argument(
        "--run_task",
        type=int,
        default=-1,
        help="Run a specific molecular property task. Use -1 to run all tasks.",
    )

    # =========================
    # Few-shot task settings
    # =========================
    parser.add_argument("--n_shot_train", type=int, default=10)
    parser.add_argument("--n_shot_test", type=int, default=10)
    parser.add_argument("--n_query", type=int, default=16)
    parser.add_argument(
        "--n_way",
        type=int,
        default=2,
        help="Few-shot molecular property prediction is formulated as 2-way K-shot classification.",
    )

    # =========================
    # Meta-learning settings
    # =========================
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--meta_batch_size", type=int, default=9)

    parser.add_argument(
        "--inner_lr",
        type=float,
        default=0.1,
        help="Learning rate for inner optimization of atom-level attention parameters.",
    )
    parser.add_argument(
        "--outer_lr",
        type=float,
        default=0.001,
        help="Learning rate for outer optimization of all HMAN parameters.",
    )
    parser.add_argument("--weight_decay", type=float, default=5e-5)

    parser.add_argument(
        "--inner_steps",
        type=int,
        default=1,
        help="Number of inner-loop update steps on support set.",
    )
    parser.add_argument(
        "--outer_steps",
        type=int,
        default=1,
        help="Number of outer-loop update steps on query set.",
    )
    parser.add_argument(
        "--test_steps",
        type=int,
        default=1,
        help="Number of adaptation steps during meta-testing.",
    )
    parser.add_argument(
        "--second_order",
        action="store_true",
        help="Use second-order gradients in meta-learning.",
    )

    # =========================
    # GNN feature extraction
    # =========================
    parser.add_argument(
        "--gnn_type",
        type=str,
        default="gin",
        choices=["gin", "gcn", "graphsage", "gat"],
        help="Backbone molecular graph encoder.",
    )
    parser.add_argument("--num_gnn_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "sum", "max"],
        help="Pooling function for molecular and prototype features.",
    )
    parser.add_argument(
        "--use_pretrained_gnn",
        action="store_true",
        default=True,
        help="Use pretrained GIN parameters for molecular representation learning.",
    )
    parser.add_argument(
        "--pretrained_gnn_path",
        type=str,
        default="./chem_lib/model_gin/supervised_contextpred.pth",
        help="Path to pretrained GNN weights.",
    )

    # =========================
    # HMAN-specific settings
    # =========================
    parser.add_argument(
        "--top_b_atoms",
        type=int,
        default=5,
        help="Number of top-ranked atoms selected by atom-level attention.",
    )
    parser.add_argument(
        "--top_k_substructures",
        type=int,
        default=3,
        help="Number of top-ranked substructures selected by molecule-level attention.",
    )
    parser.add_argument(
        "--attention_dim",
        type=int,
        default=300,
        help="Hidden dimension used in hierarchical attention modules.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Weight coefficient for weighted negative log-likelihood loss.",
    )
    parser.add_argument(
        "--num_substructure_samples",
        type=int,
        default=5,
        help="Number of sampled atom combinations used for weighted negative log-likelihood loss.",
    )

    # =========================
    # Training and evaluation
    # =========================
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--save_logs", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--result_dir", type=str, default="./results")

    # =========================
    # Device and reproducibility
    # =========================
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()

    if args.test_dataset == args.dataset:
        args.test_dataset = None

    args.device = (
        f"cuda:{args.gpu_id}"
        if torch.cuda.is_available()
        else "cpu"
    )

    return args


def main():
    args = get_args()
    set_seed(args.seed)

    exp_dir = create_experiment_dir(args.result_dir, args.dataset)
    logger = set_logger(exp_dir)
    save_config(args, exp_dir)

    logger.info("=" * 80)
    logger.info("HMAN: Hierarchical Molecular Attention Network")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Test dataset: {args.test_dataset}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Experiment directory: {exp_dir}")

    # =========================
    # Build few-shot tasks
    # =========================
    train_tasks, valid_tasks, test_tasks = build_few_shot_tasks(
        dataset=args.dataset,
        data_dir=args.data_dir,
        test_dataset=args.test_dataset,
        run_task=args.run_task,
        seed=args.seed,
    )

    logger.info(f"Number of training tasks: {len(train_tasks)}")
    logger.info(f"Number of validation tasks: {len(valid_tasks)}")
    logger.info(f"Number of test tasks: {len(test_tasks)}")

    # =========================
    # Build HMAN model
    # =========================
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
    ).to(args.device)

    num_params = count_model_params(model)
    logger.info(f"Trainable parameters: {num_params}")

    # =========================
    # Build trainer
    # =========================
    trainer = HMANTrainer(
        model=model,
        train_tasks=train_tasks,
        valid_tasks=valid_tasks,
        test_tasks=test_tasks,
        device=args.device,
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
        beta=args.beta,
        num_substructure_samples=args.num_substructure_samples,
        second_order=args.second_order,
        exp_dir=exp_dir,
        logger=logger,
    )

    # =========================
    # Initial evaluation
    # =========================
    logger.info("Initial evaluation before training.")
    best_valid_auc = trainer.evaluate(split="valid")
    best_test_auc = trainer.evaluate(split="test")

    logger.info(f"Initial valid AUC: {best_valid_auc:.4f}")
    logger.info(f"Initial test AUC: {best_test_auc:.4f}")

    # =========================
    # Meta-training
    # =========================
    logger.info("Start meta-training.")

    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        train_info = trainer.train_epoch(epoch)

        epoch_time = time.time() - epoch_start_time

        logger.info(
            f"Epoch [{epoch:04d}/{args.epochs}] "
            f"train_loss={train_info['loss']:.6f} "
            f"class_loss={train_info.get('class_loss', 0.0):.6f} "
            f"weighted_nll_loss={train_info.get('weighted_nll_loss', 0.0):.6f} "
            f"time={epoch_time:.2f}s"
        )

        # =========================
        # Evaluation
        # =========================
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

                logger.info(
                    f"New best model found at epoch {epoch}: "
                    f"best_valid_auc={best_valid_auc:.4f}, "
                    f"corresponding_test_auc={best_test_auc:.4f}"
                )

                if args.save_model:
                    trainer.save_model("best_model.pt")

        # =========================
        # Periodic checkpoint
        # =========================
        if args.save_model and epoch % args.save_steps == 0:
            trainer.save_model(f"checkpoint_epoch_{epoch}.pt")

    total_time = time.time() - total_start_time

    logger.info("=" * 80)
    logger.info("Training finished.")
    logger.info(f"Total training time: {total_time / 60:.2f} min")
    logger.info(f"Best valid AUC: {best_valid_auc:.4f}")
    logger.info(f"Corresponding test AUC: {best_test_auc:.4f}")
    logger.info("=" * 80)

    trainer.conclude()

    if args.save_logs:
        trainer.save_result_log()


if __name__ == "__main__":
    main()