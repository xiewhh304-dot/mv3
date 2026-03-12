import torch


class TimeDistorter:
    def __init__(self, train_distortion="identity", sample_distortion="identity"):
        self.train_distortion = train_distortion
        self.sample_distortion = sample_distortion

    def apply_distortion(self, t, mode):
        if mode == "identity":
            return t
        if mode == "cos":
            return (1 - torch.cos(t * torch.pi)) / 2
        if mode == "revcos":
            return 2 * t - (1 - torch.cos(t * torch.pi)) / 2
        if mode == "polyinc":
            return t ** 2
        if mode == "polydec":
            return 2 * t - t ** 2
        raise ValueError(f"Unknown distortion mode: {mode}")

    def train_ft(self, device):
        t = torch.rand(1, device=device)
        return self.apply_distortion(t, self.train_distortion)

    def sample_ft(self, t):
        return self.apply_distortion(t, self.sample_distortion)