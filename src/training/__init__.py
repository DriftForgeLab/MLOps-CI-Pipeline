from dataclasses import dataclass


@dataclass
class TrainingResult:
    model: object
    algorithm: str
    hyperparameters: dict
    dataset_version_id: str
    random_seed: int
    trained_at: str
    train_rows: int