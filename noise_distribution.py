import torch


class NoiseDistribution:
    def __init__(self, transition: str = "uniform", e_num_classes: int = 2):
        self.transition = transition
        self.e_num_classes = e_num_classes

        if transition == "uniform":
            e_limit = torch.ones(e_num_classes) / e_num_classes

        elif transition == "absorbfirst":
            e_limit = torch.zeros(e_num_classes)
            e_limit[0] = 1.0

        elif transition == "absorbing":
            e_limit = torch.zeros(e_num_classes)
            e_limit[-1] = 1.0

        else:
            raise ValueError(f"Unsupported transition: {transition}")

        self.limit_dist = {"E": e_limit}

    def get_limit_dist(self):
        return self.limit_dist