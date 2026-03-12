import random
import numpy as np
import torch
import torch.nn.functional as F

from config import ProjectConfig
from flow_matching.noise_distribution import NoiseDistribution
from flow_matching.time_distorter import TimeDistorter
from flow_matching.edge_flow_utils import sample_discrete_from_probs
from flow_matching.edge_rate_matrix import EdgeRateMatrixDesigner
from models.edge_defog_model import EdgeDeFoGModel
from utils.mat_io import load_pickle, save_pickle


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    cfg = ProjectConfig()
    cfg.generated_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.data.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # 1) Load training cache
    #    NOTE:
    #      Training used multiple views separately with ONE shared model.
    #      Sampling generates ONE shared graph from noise.
    # ------------------------------------------------------------
    cache = load_pickle(cfg.graph_cache_path)
    ckpt = torch.load(cfg.ckpt_dir / "shared_model.pt", map_location=device)

    shared_views = cache["views"]
    rrwp_nodes = cache["rrwp_node"]
    shared_edge_index = cache["shared_edge_index"]

    edge_index = torch.tensor(shared_edge_index, dtype=torch.long, device=device)

    noise_dist = NoiseDistribution(
        transition=cfg.defog.transition,
        e_num_classes=2,
    )
    limit_dist = noise_dist.get_limit_dist()

    time_distorter = TimeDistorter(
        train_distortion=cfg.defog.train_time_distortion,
        sample_distortion=cfg.defog.sample_time_distortion,
    )

    rate_designer = EdgeRateMatrixDesigner(
        eta=cfg.defog.eta,
        omega=cfg.defog.omega,
    )

    model = EdgeDeFoGModel(
        node_feat_dim=ckpt["node_feat_dim"],
        edge_num_classes=ckpt["edge_num_classes"],
        rrwp_dim=ckpt["rrwp_dim"],
        hidden_dim=ckpt.get("hidden_dim", cfg.defog.hidden_dim),
        edge_hidden_dim=ckpt.get("edge_hidden_dim", 16),
        dropout=ckpt.get("dropout", cfg.defog.dropout),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    N = shared_views[0].shape[0]
    M = edge_index.shape[0]
    if M == 0:
        raise RuntimeError("Shared candidate-edge space is empty.")

    # ------------------------------------------------------------
    # 2) Sampling starts from noise, following original DeFoG spirit:
    #      E_0 ~ p_0^E
    # ------------------------------------------------------------
    init_prob = limit_dist["E"].to(device).view(1, -1).expand(M, -1)
    e_label = sample_discrete_from_probs(init_prob)
    e_t = F.one_hot(e_label, num_classes=2).float()

    # Pre-load all views. We do NOT sample "for a specific view".
    # Instead, we use the SAME shared model and aggregate predictions from all views
    # to produce ONE shared clean graph.
    X_views = [torch.tensor(v, dtype=torch.float32, device=device) for v in shared_views]
    R_views = [torch.tensor(r, dtype=torch.float32, device=device) for r in rrwp_nodes]

    with torch.no_grad():
        for step in range(cfg.defog.sample_steps):
            t = torch.tensor([step / cfg.defog.sample_steps], dtype=torch.float32, device=device)
            s = torch.tensor([(step + 1) / cfg.defog.sample_steps], dtype=torch.float32, device=device)

            t_d = time_distorter.sample_ft(t)
            s_d = time_distorter.sample_ft(s)
            dt = float((s_d - t_d).item())

            # --------------------------------------------------------
            # 3) Use ONE shared model on ALL views separately, then
            #    aggregate the clean marginal predictions:
            #
            #    p_bar(E_1 | E_t, t) = (1/V) sum_v p_theta(E_1 | E_t, X^(v), R^(v), t)
            # --------------------------------------------------------
            pred_prob_sum = 0.0
            for X, R in zip(X_views, R_views):
                pred_logits_v = model(
                    X=X,
                    edge_index=edge_index,
                    e_t=e_t,
                    rrwp_node=R,
                    t=t_d,
                )
                pred_prob_v = F.softmax(pred_logits_v, dim=-1)
                pred_prob_sum = pred_prob_sum + pred_prob_v

            pred_e = pred_prob_sum / len(X_views)

            # final step: directly decode the clean label
            if step == cfg.defog.sample_steps - 1:
                e_label = pred_e.argmax(dim=-1)
                e_t = F.one_hot(e_label, num_classes=2).float()
            else:
                # ----------------------------------------------------
                # 4) CTMC / rate-matrix update
                #    Original DeFoG Eq.(2) spirit:
                #      p_{t+dt|t}(z_{t+dt}|z_t) = delta + R_t dt
                # ----------------------------------------------------
                R_E = rate_designer.compute_edge_rate_matrix(
                    t=t_d,
                    e_t=e_t,
                    pred_e=pred_e,
                    limit_dist=limit_dist,
                )
                prob_e = rate_designer.compute_step_probs(R_E, e_t, dt)
                e_label = sample_discrete_from_probs(prob_e)
                e_t = F.one_hot(e_label, num_classes=2).float()

    # ------------------------------------------------------------
    # 5) Decode ONE shared clean graph
    # ------------------------------------------------------------
    final_adj = torch.zeros((N, N), dtype=torch.int64, device=device)
    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    pred_lab = e_t.argmax(dim=-1).long()

    final_adj[src, dst] = pred_lab
    final_adj[dst, src] = pred_lab
    final_adj.fill_diagonal_(0)

    final_adj_np = final_adj.cpu().numpy().astype("int64")
    pred_lab_np = pred_lab.cpu().numpy().astype("int64")

    save_pickle(
        {
            "shared_adj": final_adj_np,
            "candidate_edge_labels": pred_lab_np,
            "shared_edge_index": shared_edge_index,
            "labels": cache["labels"],
        },
        cfg.generated_dir / "generated_graph.pkl",
    )

    torch.save(torch.tensor(final_adj_np), cfg.generated_dir / "shared_edge.pt")
    print("\nOne shared graph is generated from noise and saved.")


if __name__ == "__main__":
    main()
