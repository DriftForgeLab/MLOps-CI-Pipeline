# =============================================================================
# src/config/schema.py — Frozen dataclasses and validation constants
# =============================================================================
# Pure data definitions. No logic, no I/O, no project imports beyond stdlib.
# Consumed by validation.py and loader.py.
# =============================================================================

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------

VALID_TASK_TYPES: frozenset[str] = {"classification", "regression", "image_classification", "image_classification_cnn"} #!!! May need to update VALIDATE and REQUIRED to ENUM in later sprints
IMAGE_TASK_TYPES: frozenset[str] = frozenset({"image_classification", "image_classification_cnn"})
CLASSIFICATION_TASK_TYPES: frozenset[str] = frozenset({"classification"}) | IMAGE_TASK_TYPES
SKLEARN_TASK_TYPES: frozenset[str] = frozenset({"classification", "regression", "image_classification"})
VALID_LOG_LEVELS: frozenset[str] = {"DEBUG", "INFO", "WARNING", "ERROR"}
VALID_PIPELINE_STAGES: frozenset[str] = {"preprocessing", "training", "evaluation", "deployment", "promotion"} #!!! May need to update the validation of Deploymeny in later sprints
VALID_ALGORITHMS: frozenset[str] = {"random_forest", "logistic_regression", "linear_regression", "cnn"}
VALID_SOLVERS: frozenset[str] = {"lbfgs", "saga", "liblinear"}
VALID_CLASS_WEIGHTS: frozenset[str] = {"balanced"} ## May need other weights later

REQUIRED_MODEL_KEYS: frozenset[str] = {"algorithm", "hyperparameters"}
_OPTIONAL_MODEL_KEYS: frozenset[str] = frozenset({"architecture"})
RF_HYPERPARAMS_KEYS: frozenset[str] = {"n_estimators", "max_depth", "min_samples_split", "class_weight"}
LR_HYPERPARAMS_KEYS: frozenset[str] = {"C", "solver", "max_iter", "class_weight"}
CNN_HYPERPARAMS_KEYS: frozenset[str] = {"epochs", "batch_size", "learning_rate"}
CNN_ARCHITECTURE_KEYS: frozenset[str] = {"conv_layers", "fc_units", "dropout"}
CNN_CONV_LAYER_KEYS: frozenset[str] = {"out_channels", "kernel_size"}

REQUIRED_TOP_LEVEL_KEYS: frozenset[str] = {"project", "task_type", "random_seed", "pipeline_stages", "output_dir", "data", "configs", "log_level", "dataset"}

# ---------------------------------------------------------------------------
# Preprocessing constants
# ---------------------------------------------------------------------------

VALID_ENCODING_STRATEGIES: frozenset[str] = {"onehot", "ordinal"}
VALID_HANDLE_UNKNOWN: frozenset[str] = {"ignore", "error"}
VALID_SCALING_STRATEGIES: frozenset[str] = {"standard", "minmax"}
VALID_MISSING_POLICIES: frozenset[str] = {"passthrough", "fail", "impute"}
VALID_NUMERIC_IMPUTE_STRATEGIES: frozenset[str] = {"mean", "median", "constant"}
VALID_CATEGORICAL_IMPUTE_STRATEGIES: frozenset[str] = {"most_frequent", "constant"}
VALID_IMAGE_COLOR_MODES: frozenset[str] = {"rgb", "grayscale"}

REQUIRED_PROJECT_KEYS: frozenset[str] = {"name", "version"}
REQUIRED_DATA_KEYS: frozenset[str] = {"raw", "processed", "evaluation", "drift_scenarios"}
REQUIRED_CONFIGS_KEYS: frozenset[str] = {"preprocessing", "training", "evaluation", "deployment", "promotion"}

_PREPROCESSING_TOP_LEVEL_KEYS: set[str] = {
    "fail_on_nulls", "min_rows", "validate_types", "validate_labels", "validate_on_skip",
    "numeric_features", "categorical_features",
    "encoding", "scaling", "missing_values",
    "image",
}
_IMAGE_KEYS: set[str] = {"target_size", "color_mode", "normalize", "flatten", "augmentation"}
_IMAGE_AUGMENTATION_KEYS: set[str] = {"enabled", "horizontal_flip", "rotation_degrees", "augmentation_factor"}
_ENCODING_KEYS: set[str] = {"enabled", "strategy", "handle_unknown", "min_frequency"}
_SCALING_KEYS: set[str] = {"enabled", "strategy"}
_MISSING_VALUES_KEYS: set[str] = {"policy", "numeric_strategy", "categorical_strategy", "fill_value"}

# ---------------------------------------------------------------------------
# Promotion constants
# ---------------------------------------------------------------------------

VALID_OPERATORS = {">=", "<=", ">", "<", "=="}
_REQUIRED_RULE_KEYS: frozenset[str] = frozenset({"id", "metric", "threshold", "operator"})

# ---------------------------------------------------------------------------
# Deployment constants
# ---------------------------------------------------------------------------

VALID_SERVER_LOG_LEVELS: frozenset[str] = frozenset({"debug", "info", "warning", "error"})
VALID_MODEL_STAGES: frozenset[str] = frozenset({"Production", "Staging", "Archived", "None"})


# =============================================================================
# Frozen dataclasses
# =============================================================================

# ---------------------------------------------------------------------------
# Pipeline config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectMeta:
    name: str
    version: str

@dataclass(frozen=True)
class DataPaths:
    raw: str
    processed: str
    evaluation: str
    drift_scenarios: str

@dataclass(frozen=True)
class SubConfigPaths:
    preprocessing: str
    training: str
    evaluation: str
    promotion: str
    deployment: str

@dataclass(frozen=True)
class RandomForestHyperparams: ### Remember to change this DataCalss if Training_Classification.yaml is updated in later sprints
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    class_weight: str | None = None

@dataclass(frozen=True)
class LogisticRegressionHyperparams: ### Remember to change this DataCalss if Training_Regression.yaml is updated in later sprints
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 200
    class_weight: str | None = None

@dataclass(frozen=True)
class LinearRegressionHyperparams:
    pass

@dataclass(frozen=True)
class CnnConvLayerConfig:
    out_channels: int
    kernel_size: int

@dataclass(frozen=True)
class CnnArchitectureConfig:
    conv_layers: tuple[CnnConvLayerConfig, ...]
    fc_units: int
    dropout: float

@dataclass(frozen=True)
class CnnHyperparams:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001

@dataclass(frozen=True)
class ModelConfig:
    algorithm: str
    hyperparameters: RandomForestHyperparams | LogisticRegressionHyperparams | LinearRegressionHyperparams | CnnHyperparams
    architecture: CnnArchitectureConfig | None = None

@dataclass(frozen=True)
class TrainingConfig:
    model: ModelConfig

# ---------------------------------------------------------------------------
# Preprocessing config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EncodingConfig:
    enabled: bool = False
    strategy: str = "onehot"
    handle_unknown: str = "ignore"
    min_frequency: int | None = None

@dataclass(frozen=True)
class ScalingConfig:
    enabled: bool = False
    strategy: str = "standard"

@dataclass(frozen=True)
class MissingValuesConfig:
    policy: str = "passthrough"
    numeric_strategy: str = "mean"
    categorical_strategy: str = "most_frequent"
    fill_value: object = None

@dataclass(frozen=True)
class ImageAugmentationConfig:
    enabled: bool = False
    horizontal_flip: bool = False
    rotation_degrees: int = 0
    augmentation_factor: int = 1  # number of augmented copies per original

@dataclass(frozen=True)
class ImagePreprocessingConfig:
    target_size: tuple[int, int] = (64, 64)  # (height, width)
    color_mode: str = "rgb"                   # "rgb" or "grayscale"
    normalize: bool = True                     # scale pixels to [0,1]
    flatten: bool = True                       # flatten to 1D for sklearn
    augmentation: ImageAugmentationConfig = field(default_factory=ImageAugmentationConfig)

@dataclass(frozen=True)
class PreprocessingConfig:
    # Data contract enforcement
    fail_on_nulls: bool = True
    min_rows: int = 10
    validate_types: bool = True
    validate_labels: bool = True
    validate_on_skip: bool = False
    # Feature type overrides (None = auto-detect from schema)
    numeric_features: tuple[str, ...] | None = None
    categorical_features: tuple[str, ...] | None = None
    # Transform policies
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    missing_values: MissingValuesConfig = field(default_factory=MissingValuesConfig)
    # Image preprocessing (None = not an image dataset)
    image: ImagePreprocessingConfig | None = None

@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str | None = None
    experiment_name: str | None = None
    registry_model_name: str | None = None

@dataclass(frozen=True)
class PipelineConfig:
    project: ProjectMeta
    task_type: str
    random_seed: int
    pipeline_stages: tuple[str, ...]
    output_dir: str
    data: DataPaths
    configs: SubConfigPaths
    log_level: str
    dataset: str
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

# ---------------------------------------------------------------------------
# Evaluation config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassificationEvalConfig:
    averaging: str = "weighted"

@dataclass(frozen=True) ### This dataclass is currently empty
class RegressionEvalConfig:
    pass

@dataclass(frozen=True)
class EvaluationConfig:
    classification: ClassificationEvalConfig = field(default_factory=ClassificationEvalConfig)
    regression: RegressionEvalConfig = field(default_factory=RegressionEvalConfig)

# ---------------------------------------------------------------------------
# Promotion config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromotionRule:
    id: str
    metric: str
    threshold: float
    operator: str
    description: str

@dataclass(frozen=True)
class PromotionTaskConfig:
    rules: tuple[PromotionRule, ...]

@dataclass(frozen=True)
class PromotionConfig:
    classification: PromotionTaskConfig
    regression: PromotionTaskConfig

# ---------------------------------------------------------------------------
# Deployment config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

@dataclass(frozen=True)
class ModelServingConfig:
    allowed_stage: str = "Production"
    require_production_model: bool = True
    startup_timeout_seconds: int = 120

@dataclass(frozen=True)
class HealthcheckConfig:
    include_model_info: bool = True

@dataclass(frozen=True)
class DeploymentConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelServingConfig = field(default_factory=ModelServingConfig)
    healthcheck: HealthcheckConfig = field(default_factory=HealthcheckConfig)
