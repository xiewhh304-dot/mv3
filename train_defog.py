import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from config import ProjectConfig
from flow_matching.noise_distribution import NoiseDistribution
from flow_matching.time_distorter import TimeDistorter
from flow_matching.edge_flow_utils import (
    sample_noisy_edge_vector_from_clean,
    edge_vector_to_onehot,
)
from models.edge_defog_model import EdgeDeFoGModel
from utils.mat_io import load_multiview_mat, save_pickle
from utils.graph_builder import build_multiview_graphs, sample_negative_edges
from utils.rrwp import compute_rrwp


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Loss metrics
#   Two choices:
#   1) Cross Entropy
#   2) KL Divergence
# Inspired by:
#   if not kld:
#       self.node_loss = CrossEntropyMetric()
#       self.edge_loss = CrossEntropyMetric()
#   else:
#       self.node_loss = KLDMetric()
#       self.edge_loss = KLDMetric()
# =========================
class CrossEntropyMetric(nn.Module):
    """
    For logits pred:[M,K] and labels target:[M]
    L_CE = - sum_m log p_theta(e_1^{(m)} | e_t^{(m)}, ...)
    """
    def forward(self, pred_logits: torch.Tensor, target_label: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(pred_logits, target_label)


class KLDMetric(nn.Module):
    """
    For logits pred:[M,K] and one-hot target prob:[M,K]
    L_KL = sum_m KL( q(e_1^{(m)}) || p_theta(e_1^{(m)} | e_t^{(m)}, ...) )
    Here q is the one-hot clean target distribution.
    """
    def forward(self, pred_logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(pred_logits, dim=-1)
        return F.kl_div(log_p, target_prob, reduction="batchmean")


def build_shared_candidate_space(adjs, seed=42, min_view_count=2):
    """
    共享 candidate-edge 空间:
      正边: 至少在 min_view_count 个视图中出现
      负边: 从所有视图都无边的位置随机采样，数量与正边相同
    """
    num_views = len(adjs)
    N = adjs[0].shape[0]

    edge_count = np.zeros((N, N), dtype=np.int64)
    for A in adjs:
        edge_count += np.triu(A, k=1).astype(np.int64)

    pos_mask = edge_count >= min_view_count
    pos_row, pos_col = np.where(pos_mask)
    pos_edges = np.stack([pos_row, pos_col], axis=1).astype(np.int64)

    # 所有视图都无边的位置
    zero_mask = edge_count == 0
    zero_full = zero_mask.astype(np.int64)
    zero_full = zero_full + zero_full.T

    neg_edges = sample_negative_edges(zero_full, num_neg=len(pos_edges), seed=seed)
    if neg_edges.ndim == 1:
        neg_edges = neg_edges.reshape(-1, 2)

    if len(neg_edges) == 0:
        edge_index = pos_edges
    else:
        edge_index = np.concatenate([pos_edges, neg_edges], axis=0).astype(np.int64)

    edge_labels_per_view = []
    for A in adjs:
        labels = A[edge_index[:, 0], edge_index[:, 1]].astype(np.int64)
        edge_labels_per_view.append(labels)

    return edge_index, edge_labels_per_view


def pad_or_truncate_features(X: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Shared model requires a fixed node feature dimension.
    We DO NOT fuse views at training time.
    Instead, each view is separately fed into the SAME model after:
        X^(v) -> \tilde{X}^(v) in R^{N x d_shared}
    """
    n, d = X.shape
    if d == target_dim:
        return X.astype(np.float32)
    if d > target_dim:
        return X[:, :target_dim].astype(np.float32)
    out = np.zeros((n, target_dim), dtype=np.float32)
    out[:, :d] = X.astype(np.float32)
    return out


def main():
    cfg = ProjectConfig()
    cfg.dataset_output_dir.mkdir(parents=True, exist_ok=True)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.data.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # loss switch:
    #   export DEFOG_EDGE_LOSS=ce
    #   export DEFOG_EDGE_LOSS=kld
    loss_type = os.getenv("DEFOG_EDGE_LOSS", "ce").lower()
    use_kld = (loss_type == "kld")

    if not use_kld:
        edge_loss_fn = CrossEntropyMetric()
    else:
        edge_loss_fn = KLDMetric()

    # ------------------------------------------------------------
    # 1) Load multi-view data
    #    {X^(v)}_{v=1}^V , labels Y
    # ------------------------------------------------------------
    views, labels = load_multiview_mat(
        str(cfg.mat_path),
        x_key=cfg.data.x_key,
        y_key=cfg.data.y_key,
    )

    # ------------------------------------------------------------
    # 2) Build one KNN graph per view:
    #    X^(v) -> A^(v)
    # ------------------------------------------------------------
    proc_views, adjs = build_multiview_graphs(
        views,
        k=cfg.data.knn_k,
        metric=cfg.data.graph_metric,
        sym_mode=cfg.data.sym_mode,
    )

    # ------------------------------------------------------------
    # 3) RRWP structural features per view
    #    R^(v) = RRWP(A^(v))
    # ------------------------------------------------------------
    rrwp_nodes = []
    for A in adjs:
        node_rrwp, _ = compute_rrwp(
            A,
            walk_length=cfg.rrwp.walk_length,
            add_identity=cfg.rrwp.add_identity,
        )
        rrwp_nodes.append(node_rrwp.astype(np.float32))

    # ------------------------------------------------------------
    # 4) Shared candidate-edge space across all views
    #    Positive: union of view edges
    #    Negative: random non-edges
    # ------------------------------------------------------------
    shared_edge_index, edge_labels_per_view = build_shared_candidate_space(
        adjs,
        seed=cfg.data.seed,
    )
    if len(shared_edge_index) == 0:
        raise RuntimeError("Shared candidate-edge space is empty.")

    # ------------------------------------------------------------
    # 5) Shared feature dimension for one shared model
    #    Views are NOT fused in training.
    #    They are separately fed to the SAME model.
    # ------------------------------------------------------------
    shared_in_dim = max(x.shape[1] for x in proc_views)
    shared_views = [pad_or_truncate_features(x, shared_in_dim) for x in proc_views]

    # Save cache for sampling + classifier
    save_pickle(
        {
            "views": shared_views,              # per-view aligned features (NOT fused)
            "raw_views": proc_views,
            "edge_labels": adjs,
            "rrwp_node": rrwp_nodes,
            "shared_edge_index": shared_edge_index,
            "edge_labels_per_view": edge_labels_per_view,
            "labels": labels,
            "shared_in_dim": shared_in_dim,
        },
        cfg.graph_cache_path,
    )

    save_pickle(
        {
            "num_views": len(shared_views),
            "shared_in_dim": shared_in_dim,
            "rrwp_dim": rrwp_nodes[0].shape[1],
            "edge_num_states": 2,
            "loss_type": loss_type,
        },
        cfg.meta_path,
    )

    # ------------------------------------------------------------
    # 6) Define DeFoG noising components
    #    Original DeFoG Eq.(1) edge-only version:
    #      p_{t|1}(e_t | e_1) = t * delta(e_t, e_1) + (1-t) * p_0^E(e_t)
    # ------------------------------------------------------------
    noise_dist = NoiseDistribution(
        transition=cfg.defog.transition,
        e_num_classes=2,
    )
    limit_dist = noise_dist.get_limit_dist()

    time_distorter = TimeDistorter(
        train_distortion=cfg.defog.train_time_distortion,
        sample_distortion=cfg.defog.sample_time_distortion,
    )

    # ------------------------------------------------------------
    # 7) One shared model for all views
    # ------------------------------------------------------------
    model = EdgeDeFoGModel(
        node_feat_dim=shared_in_dim,
        edge_num_classes=2,
        rrwp_dim=rrwp_nodes[0].shape[1],
        hidden_dim=cfg.defog.hidden_dim,
        edge_hidden_dim=16,
        dropout=cfg.defog.dropout,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.defog.lr,
        weight_decay=cfg.defog.weight_decay,
    )

    best_loss = float("inf")
    save_path = cfg.ckpt_dir / "shared_model.pt"

    edge_index = torch.tensor(shared_edge_index, dtype=torch.long, device=device)

    # ------------------------------------------------------------
    # 8) Train the SAME model with multiple views separately
    #
    # Objective:
    #   L(theta) = sum_v L^(v)(theta)
    #
    # For each view v:
    #   - sample t
    #   - sample noisy edge states E_t^(v)
    #   - predict clean marginal p_theta(E_1^(v) | E_t^(v), X^(v), R^(v), t)
    # ------------------------------------------------------------
    for epoch in range(1, cfg.defog.epochs + 1):
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        used_views = 0

        for view_id, (X_np, rrwp_np, e1_np) in enumerate(
            zip(shared_views, rrwp_nodes, edge_labels_per_view)
        ):
            X = torch.tensor(X_np, dtype=torch.float32, device=device)
            rrwp_node = torch.tensor(rrwp_np, dtype=torch.float32, device=device)
            e1_label = torch.tensor(e1_np, dtype=torch.long, device=device)

            # sample time t (possibly distorted)
            t = time_distorter.train_ft(device)

            # noising:
            #   E_t^(v) ~ p_{t|1}( E_t^(v) | E_1^(v) )
            e_t = sample_noisy_edge_vector_from_clean(e1_label, t, limit_dist)

            # clean marginal prediction
            pred_e = model(
                X=X,
                edge_index=edge_index,
                e_t=e_t,
                rrwp_node=rrwp_node,
                t=t,
            )

            # loss choice: CE or KL
            if use_kld:
                target_prob = edge_vector_to_onehot(e1_label, num_classes=2)
                loss_e = edge_loss_fn(pred_e, target_prob)
            else:
                loss_e = edge_loss_fn(pred_e, e1_label)

            total_loss = total_loss + cfg.defog.lambda_edge * loss_e
            used_views += 1

        loss = total_loss / max(used_views, 1)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_loss": best_loss,
                    "node_feat_dim": shared_in_dim,
                    "rrwp_dim": rrwp_nodes[0].shape[1],
                    "edge_num_classes": 2,
                    "hidden_dim": cfg.defog.hidden_dim,
                    "edge_hidden_dim": 16,
                    "dropout": cfg.defog.dropout,
                    "seed": cfg.data.seed,
                    "shared_model": True,
                    "loss_type": loss_type,
                },
                save_path,
            )

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"[Shared DeFoG] Epoch {epoch:03d} | "
                f"loss={loss.item():.6f} | best={best_loss:.6f} | "
                f"edge_loss={loss_type}"
            )

    print("\nShared multi-view DeFoG model is saved.")


if __name__ == "__main__":
    main()
