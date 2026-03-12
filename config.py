from dataclasses import dataclass, field
from pathlib import Path
import os

@dataclass
class DataConfig:
    # data_root: str = "D:/mvdata"
    # dataset_name: str = "animals"   # 不写 .mat
    data_root: str = os.getenv("DATA_ROOT", "D:/mvdata")
    # data_root: str = os.getenv("DATA_ROOT", "/public/home/wsp/inspur/dhy/mvdata")
    dataset_name: str = os.getenv("DATASET_NAME", "scene15")  # 不写 .mat
    x_key: str = "X"
    y_key: str = "Y"

    train_ratio: float = 0.1
    test_ratio: float = 0.9
    # seed: int = 42
    seed: int = int(os.getenv("SEED", 42))
    # graph construction
    knn_k: int = 10
    graph_metric: str = "cosine"     # cosine / euclidean
    sym_mode: str = "or"             # or / and


@dataclass
class RRWPConfig:
    walk_length: int = 8             # RRWP 最大步长
    add_identity: bool = True        # 是否包含 0-step / identity


@dataclass
class DefogConfig:
    hidden_dim: int = 64
    dropout: float = 0.1

    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 200

    lambda_edge: float = 1.0

    # noise / transition
    transition: str = "uniform"      # uniform / absorbfirst / absorbing

    # sampling
    sample_steps: int = 50
    train_time_distortion: str = "identity"
    sample_time_distortion: str = "identity"

    omega: float = 0.0               # target guidance
    eta: float = 0.0                 # stochasticity


@dataclass
class ClassifierConfig:
    hidden_dim: int = 64
    out_dim: int = 64
    dropout: float = 0.5
    lr: float = 1e-4
    weight_decay: float = 5e-4
    epochs: int = 300


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    rrwp: RRWPConfig = field(default_factory=RRWPConfig)
    defog: DefogConfig = field(default_factory=DefogConfig)
    clf: ClassifierConfig = field(default_factory=ClassifierConfig)

    # output_root: str = "./outputs"
    output_root: str = os.getenv("OUTPUT_ROOT", "./outputs")
    @property
    def mat_path(self) -> Path:
        return Path(self.data.data_root) / f"{self.data.dataset_name}.mat"

    @property
    def dataset_output_dir(self) -> Path:
        return Path(self.output_root) / self.data.dataset_name

    @property
    def split_path(self) -> Path:
        return self.dataset_output_dir / "splits.pkl"

    @property
    def meta_path(self) -> Path:
        return self.dataset_output_dir / "meta.pkl"

    @property
    def graph_cache_path(self) -> Path:
        return self.dataset_output_dir / "graph_cache.pkl"

    @property
    def ckpt_dir(self) -> Path:
        return self.dataset_output_dir / "defog_ckpts"

    @property
    def generated_dir(self) -> Path:
        return self.dataset_output_dir / "generated_graphs"