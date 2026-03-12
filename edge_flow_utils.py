import torch
import torch.nn.functional as F


def edge_labels_to_onehot(E, num_classes=2):
    return F.one_hot(E.long(), num_classes=num_classes).float()


def p_et_g_e1(E1, t, limit_dist):
    """
    线性 noising:
        p_{t|1}(e_t | e_1) = t * delta(e_t, e_1) + (1-t) * p0(e_t)

    输入:
        E1: [N, N] int64
        t:  [1]
        limit_dist["E"]: [K]

    返回:
        prob_Et: [N, N, K]
    """
    e_limit = limit_dist["E"].to(E1.device)
    Ke = len(e_limit)

    E1_onehot = F.one_hot(E1, num_classes=Ke).float()
    t_edge = t.view(1, 1, 1)

    Et = t_edge * E1_onehot + (1.0 - t_edge) * e_limit.view(1, 1, Ke)
    return Et.clamp(min=0.0, max=1.0)


def dt_p_et_g_e1(E1, limit_dist):
    """
    时间导数:
        d/dt p_{t|1}(e_t | e_1) = onehot(e_1) - p0
    """
    e_limit = limit_dist["E"].to(E1.device)
    Ke = len(e_limit)

    E1_onehot = F.one_hot(E1, num_classes=Ke).float()
    dE = E1_onehot - e_limit.view(1, 1, Ke)
    return dE


def sample_discrete_from_probs(prob):
    """
    prob: [..., K]
    return: [...]
    """
    flat = prob.reshape(-1, prob.shape[-1])
    sample = torch.multinomial(flat, 1).squeeze(-1)
    return sample.reshape(prob.shape[:-1])


def symmetrize_edge_labels(E):
    """
    E: [N, N]
    返回无向对称图
    """
    upper = torch.triu(E, diagonal=1)
    E_sym = upper + upper.T
    E_sym.fill_diagonal_(0)
    return E_sym


def sample_noisy_edges_from_clean(E1, t, limit_dist):
    """
    从 clean edge labels 采样 noisy edges
    输入:
        E1: [N, N]
    返回:
        E_t_onehot: [N, N, K]
    """
    prob_Et = p_et_g_e1(E1, t, limit_dist)
    Et_label = sample_discrete_from_probs(prob_Et)
    Et_label = symmetrize_edge_labels(Et_label)
    E_t = edge_labels_to_onehot(Et_label, num_classes=prob_Et.shape[-1])
    return E_t

def edge_vector_to_onehot(e, num_classes=2):
    """
    e: [M]
    return: [M, K]
    """
    return F.one_hot(e.long(), num_classes=num_classes).float()


def p_et_g_e1_vec(e1, t, limit_dist):
    """
    e1: [M]
    t: [1]
    return: [M, K]
    """
    e_limit = limit_dist["E"].to(e1.device)
    Ke = len(e_limit)

    e1_onehot = F.one_hot(e1, num_classes=Ke).float()
    prob = t.view(1, 1) * e1_onehot + (1.0 - t.view(1, 1)) * e_limit.view(1, Ke)
    return prob


def dt_p_et_g_e1_vec(e1, limit_dist):
    """
    e1: [M]
    return: [M, K]
    """
    e_limit = limit_dist["E"].to(e1.device)
    Ke = len(e_limit)

    e1_onehot = F.one_hot(e1, num_classes=Ke).float()
    return e1_onehot - e_limit.view(1, Ke)


def sample_noisy_edge_vector_from_clean(e1, t, limit_dist):
    """
    e1: [M]
    return: e_t_onehot: [M, K]
    """
    prob = p_et_g_e1_vec(e1, t, limit_dist)
    e_t = sample_discrete_from_probs(prob)
    e_t = edge_vector_to_onehot(e_t, num_classes=prob.shape[-1])
    return e_t