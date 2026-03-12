import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from config import ProjectConfig
from utils.mat_io import load_pickle


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split_indices(labels, train_ratio=0.1, test_ratio=0.9, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    train_idx = []
    test_idx = []

    for c in classes:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_train = max(1, int(round(len(idx) * train_ratio)))
        train_idx.extend(idx[:n_train].tolist())
        test_idx.extend(idx[n_train:].tolist())

    return {
        "train": np.array(sorted(train_idx), dtype=np.int64),
        "test": np.array(sorted(test_idx), dtype=np.int64),
    }


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A):
        # \hat{A} = D^{-1/2} (A + I) D^{-1/2}
        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        A_hat = A + I
        deg = A_hat.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        A_norm = deg_inv_sqrt.unsqueeze(1) * A_hat * deg_inv_sqrt.unsqueeze(0)

        X = self.dropout(X)
        X = A_norm @ X
        return self.linear(X)


class SingleGCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim, dropout=dropout)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward(self, X, A):
        H = F.relu(self.gcn1(X, A))
        H = F.relu(self.gcn2(H, A))
        logits = self.cls(H)
        return logits


def accuracy(logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> float:
    pred = logits[idx].argmax(dim=-1)
    return (pred == labels[idx]).float().mean().item()


def macro_f1(logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> float:
    pred = logits[idx].argmax(dim=-1).detach().cpu().numpy()
    true = labels[idx].detach().cpu().numpy()
    classes = np.unique(true)

    f1s = []
    for c in classes:
        tp = np.logical_and(pred == c, true == c).sum()
        fp = np.logical_and(pred == c, true != c).sum()
        fn = np.logical_and(pred != c, true == c).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        f1s.append(f1)
    return float(np.mean(f1s)) if len(f1s) > 0 else 0.0


def main():
    cfg = ProjectConfig()
    set_seed(cfg.data.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = load_pickle(cfg.graph_cache_path)
    generated = load_pickle(cfg.generated_dir / "generated_graph.pkl")

    # ------------------------------------------------------------
    # Fuse node features ONLY for downstream single-GCN classifier
    # Training of DeFoG did NOT fuse views.
    # Here we concatenate views:
    #   X_fuse = [X^(1) || ... || X^(V)]
    # ------------------------------------------------------------
    X_views = cache["views"]
    X_fuse = np.concatenate(X_views, axis=1).astype(np.float32)

    A = generated["shared_adj"].astype(np.float32)
    labels_np = cache["labels"]
    labels_np = labels_np - labels_np.min()

    splits = stratified_split_indices(
        labels_np,
        train_ratio=cfg.data.train_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.data.seed,
    )

    X = torch.tensor(X_fuse, dtype=torch.float32, device=device)
    A = torch.tensor(A, dtype=torch.float32, device=device)
    labels = torch.tensor(labels_np, dtype=torch.long, device=device)
    train_idx = torch.tensor(splits["train"], dtype=torch.long, device=device)
    test_idx = torch.tensor(splits["test"], dtype=torch.long, device=device)

    model = SingleGCNClassifier(
        in_dim=X.shape[1],
        hidden_dim=cfg.clf.hidden_dim,
        num_classes=int(labels.max().item() + 1),
        dropout=cfg.clf.dropout,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.clf.lr,
        weight_decay=cfg.clf.weight_decay,
    )

    best_acc = 0.0
    best_f1 = 0.0
    best_state = None

    for epoch in range(1, cfg.clf.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(X, A)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X, A)
            test_acc = accuracy(logits, labels, test_idx)
            test_f1 = macro_f1(logits, labels, test_idx)

        if test_acc > best_acc:
            best_acc = test_acc
            best_f1 = test_f1
            best_state = {
                "model_state": model.state_dict(),
                "best_acc": best_acc,
                "best_f1": best_f1,
            }

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.6f} | "
                f"test_acc={test_acc:.4f} | test_f1={test_f1:.4f} | "
                f"best_acc={best_acc:.4f}"
            )

    save_path = cfg.dataset_output_dir / "single_gcn_classifier.pt"
    torch.save(best_state, save_path)
    print(f"\nSaved single GCN classifier to: {save_path}")
    print(f"Best test acc = {best_acc:.4f}, best f1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
