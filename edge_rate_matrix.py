import torch
import torch.nn.functional as F

from flow_matching.edge_flow_utils import (
    p_et_g_e1_vec,
    dt_p_et_g_e1_vec,
    sample_discrete_from_probs,
)


class EdgeRateMatrixDesigner:
    def __init__(self, eta=0.0, omega=0.0):
        self.eta = eta
        self.omega = omega

    def compute_dfm_variables(self, t, e_t_label, e1_sampled, limit_dist):
        """
        e_t_label: [M]
        e1_sampled: [M]
        """
        dE = dt_p_et_g_e1_vec(e1_sampled, limit_dist)        # [M,K]
        pE = p_et_g_e1_vec(e1_sampled, t, limit_dist)        # [M,K]

        dE_at = dE.gather(-1, e_t_label.unsqueeze(-1)).squeeze(-1)  # [M]
        pE_at = pE.gather(-1, e_t_label.unsqueeze(-1)).squeeze(-1)  # [M]

        Ze = torch.count_nonzero(pE, dim=-1)                        # [M]

        return {
            "dE": dE,
            "pE": pE,
            "dE_at": dE_at,
            "pE_at": pE_at,
            "Ze": Ze,
        }

    def compute_Rstar(self, vars_):
        inner = vars_["dE"] - vars_["dE_at"].unsqueeze(-1)   # [M,K]
        numer = F.relu(inner)
        denom = (vars_["Ze"] * vars_["pE_at"]).unsqueeze(-1).clamp(min=1e-8)
        return numer / denom

    def compute_Rdb(self, vars_):
        return vars_["pE"] * self.eta

    def compute_Rtg(self, e1_sampled, e_t_label, vars_):
        Ke = vars_["pE"].shape[-1]
        e1_onehot = F.one_hot(e1_sampled, num_classes=Ke).float()

        mask = e1_sampled != e_t_label
        numer = e1_onehot * self.omega * mask.unsqueeze(-1)

        denom = (vars_["Ze"] * vars_["pE_at"]).unsqueeze(-1).clamp(min=1e-8)
        return numer / denom

    def stabilize(self, R_E, vars_):
        R_E = torch.nan_to_num(R_E, nan=0.0, posinf=0.0, neginf=0.0)
        R_E[vars_["pE"] == 0.0] = 0.0
        return R_E

    def compute_edge_rate_matrix(self, t, e_t, pred_e, limit_dist):
        """
        e_t:   [M,K]
        pred_e:[M,K]
        """
        e_t_label = e_t.argmax(dim=-1)                  # [M]
        e1_sampled = sample_discrete_from_probs(pred_e) # [M]

        vars_ = self.compute_dfm_variables(t, e_t_label, e1_sampled, limit_dist)
        Rstar = self.compute_Rstar(vars_)
        Rdb = self.compute_Rdb(vars_)
        Rtg = self.compute_Rtg(e1_sampled, e_t_label, vars_)

        R_E = self.stabilize(Rstar + Rdb + Rtg, vars_)
        return R_E

    def compute_step_probs(self, R_E, e_t, dt):
        """
        R_E: [M,K]
        e_t: [M,K]
        """
        prob = R_E * dt
        e_t_label = e_t.argmax(dim=-1)

        prob.scatter_(-1, e_t_label.unsqueeze(-1), 0.0)
        prob.scatter_(
            -1,
            e_t_label.unsqueeze(-1),
            (1.0 - prob.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )
        return prob