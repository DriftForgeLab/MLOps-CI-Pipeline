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

VALID_TASK_TYPES: frozenset[str] = frozenset({"classification", "regression", "image_classification_cnn"})
IMAGE_TASK_TYPES: frozenset[str] = frozenset({"image_classification_cnn"})
CLASSIFICATION_TASK_TYPES: frozenset[str] = frozenset({"classification"}) | IMAGE_TASK_TYPES
SKLEARN_TASK_TYPES: frozenset[str] = frozenset({"classification", "regression"})
VALID_LOG_LEVELS: frozenset[str] = {"DEBUG", "INFO", "WARNING", "ERROR"}
VALID_PIPELINE_STAGES: frozenset[str] = {"preprocessing", "training", "evaluation", "model_analysis", "deployment", "promotion"}
VALID_ALGORITHMS: frozenset[str] = {"random_forest", "logistic_regression", "linear_regression", "cnn", "gradient_boosting"}
VALID_SOLVERS: frozenset[str] = {"lbfgs", "saga", "liblinear"}
# Only sklearn's "balanced" class-weight strategy is accepted. Add more
# strategies here (and to the loaders) if other weighting schemes are needed.
VALID_CLASS_WEIGHTS: frozenset[str] = {"balanced"}

REQUIRED_MODEL_KEYS: frozenset[str] = {"algorithm", "hyperparameters"}
_OPTIONAL_MODEL_KEYS: frozenset[str] = frozenset({"architecture"})
RF_HYPERPARAMS_KEYS: frozenset[str] = {"n_estimators", "max_depth", "min_samples_split", "class_weight"}
LR_HYPERPARAMS_KEYS: frozenset[str] = {"C", "solver", "max_iter", "class_weight"}
CNN_HYPERPARAMS_KEYS: frozenset[str] = {"epochs", "batch_size", "learning_rate"}
CNN_FINE_TUNE_KEYS: frozenset[str] = {
    "enabled",
    "epochs",
    "learning_rate",
    "freeze_backbone",
    "oversample_drift_ratio",
    "use_drift_val_for_best_epoch",
    "drift_val_ratio",
}
CNN_ARCHITECTURE_KEYS: frozenset[str] = {"conv_layers", "fc_units", "dropout"}
CNN_CONV_LAYER_KEYS: frozenset[str] = {"out_channels", "kernel_size"}

REQUIRED_TOP_LEVEL_KEYS: frozenset[str] = {"project", "task_type", "random_seed", "pipeline_stages", "output_dir", "data", "configs", "log_level", "dataset", "mlflow"}

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
VALID_DEMOSAICING_ALGORITHMS: frozenset[str] = frozenset({"bilinear", "malvar2004", "menon2007"})
VALID_DENOISING_ALGORITHMS: frozenset[str] = frozenset({"gaussian", "median", "none"})
VALID_SHARPENING_ALGORITHMS: frozenset[str] = frozenset({"unsharp_mask", "sharpen_filter", "none"})

REQUIRED_PROJECT_KEYS: frozenset[str] = {"name", "version"}
REQUIRED_DATA_KEYS: frozenset[str] = {"raw", "processed", "evaluation", "drift_scenarios"}
REQUIRED_CONFIGS_KEYS: frozenset[str] = {"preprocessing", "training", "evaluation", "deployment", "promotion", "drift"}

_PREPROCESSING_TOP_LEVEL_KEYS: set[str] = {
    "fail_on_nulls", "min_rows", "validate_types", "validate_labels", "validate_on_skip",
    "numeric_features", "categorical_features",
    "encoding", "scaling", "missing_values",
    "image",
}
_IMAGE_KEYS: set[str] = {"target_size", "color_mode", "normalize", "flatten", "augmentation", "raw_input", "isp"}
_IMAGE_AUGMENTATION_KEYS: set[str] = {
    "enabled",
    "horizontal_flip",
    "rotation_degrees",
    "augmentation_factor",
    "random_crop_padding",
    "brightness_jitter",
    "contrast_jitter",
    "saturation_jitter",
}
_ISP_KEYS: set[str] = {"black_level_correction", "demosaicing", "white_balance", "color_correction", "denoising", "sharpening", "gamma_correction", "gpu_accelerated"}
_ISP_BLACK_LEVEL_KEYS: set[str] = {"enabled", "black_level"}
_ISP_DEMOSAICING_KEYS: set[str] = {"algorithm"}
_ISP_WHITE_BALANCE_KEYS: set[str] = {"r_gain", "g_gain", "b_gain"}
_ISP_COLOR_CORRECTION_KEYS: set[str] = {"enabled", "matrix"}
_ISP_DENOISING_KEYS: set[str] = {"algorithm", "strength"}
_ISP_SHARPENING_KEYS: set[str] = {"algorithm", "radius", "amount"}
_ISP_GAMMA_KEYS: set[str] = {"gamma"}
_ENCODING_KEYS: set[str] = {"enabled", "strategy", "handle_unknown", "min_frequency"}
_SCALING_KEYS: set[str] = {"enabled", "strategy"}
_MISSING_VALUES_KEYS: set[str] = {"policy", "numeric_strategy", "categorical_strategy", "fill_value"}

# ---------------------------------------------------------------------------
# Promotion constants
# ---------------------------------------------------------------------------

VALID_OPERATORS = {">=", "<=", ">", "<", "=="}
_REQUIRED_RULE_KEYS: frozenset[str] = frozenset({"id", "metric", "threshold", "operator"})

# ---------------------------------------------------------------------------
# Drift constants
# ---------------------------------------------------------------------------

VALID_DRIFT_SEVERITIES: frozenset[str] = frozenset({"low", "medium", "high"})
VALID_DRIFT_FAIL_ON_SEVERITIES: frozenset[str] = frozenset({"low", "medium", "high", "never"})
VALID_DRIFT_STATTESTS: frozenset[str] = frozenset({
    "ks", "chisquare", "psi", "wasserstein", "jensenshannon"
})
VALID_REFERENCE_SOURCES: frozenset[str] = frozenset({"train"})

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
    drift: str

# This dataclass mirrors the hyperparameter block of training_classification.yaml.
# If new Random Forest hyperparameters are added to that YAML file, add the
# matching fields here (and to RF_HYPERPARAMS_KEYS) so the loader stays in sync.
@dataclass(frozen=True)
class RandomForestHyperparams:
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    class_weight: str | None = None

# This dataclass mirrors the hyperparameter block used by the Logistic
# Regression classifier. Keep its fields aligned with LR_HYPERPARAMS_KEYS and
# the corresponding training YAML if either is extended.
@dataclass(frozen=True)
class LogisticRegressionHyperparams:
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 200
    class_weight: str | None = None

@dataclass(frozen=True)
class LinearRegressionHyperparams:
    pass

@dataclass(frozen=True)
class GradientBoostingHyperparams:
    max_iter: int = 300
    learning_rate: float = 0.05
    max_depth: int | None = 7
    min_samples_leaf: int = 20
    l2_regularization: float = 0.1

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
class FineTuneConfig:
    """Hyperparameters used when fine-tuning from existing Production model weights.

    Attributes:
        enabled:         Whether fine-tuning is active for this run.
        epochs:          Number of additional training epochs (fewer than full training).
        learning_rate:   Lower LR than full training to avoid overwriting learned features.
        freeze_backbone: If True, only the final FC classifier head is trained;
                         convolutional layers are frozen.
        oversample_drift_ratio:
                         Target fraction of drifted samples in each fine-tune batch
                         (0.0–1.0). Requires the training NPZ to carry an
                         ``is_drifted`` mask (written by the preprocessing step
                         when filenames contain a ``_drifted`` suffix). When
                         set, a ``WeightedRandomSampler`` is used so the
                         drifted minority is oversampled even if it makes up
                         only a few percent of the dataset on disk.  None
                         disables oversampling (uniform shuffling — current
                         behaviour).
        use_drift_val_for_best_epoch:
                         If True and drifted samples are present in the
                         training set, a stratified slice of those drifted
                         samples is held out as a ``drift_val`` set, excluded
                         from the training loader, and used to score
                         best-epoch selection during fine-tuning. Without
                         this flag, the standard val split is used (which is
                         dominated by clean samples and a poor proxy for
                         drift performance).
        drift_val_ratio:
                         Fraction of drifted training samples to carve out as
                         drift_val when ``use_drift_val_for_best_epoch`` is
                         True. Default 0.15.
    """
    enabled: bool = False
    epochs: int = 5
    learning_rate: float = 0.0001
    freeze_backbone: bool = False
    oversample_drift_ratio: float | None = None
    use_drift_val_for_best_epoch: bool = False
    drift_val_ratio: float = 0.15

@dataclass(frozen=True)
class ModelConfig:
    algorithm: str
    hyperparameters: RandomForestHyperparams | LogisticRegressionHyperparams | LinearRegressionHyperparams | GradientBoostingHyperparams | CnnHyperparams
    architecture: CnnArchitectureConfig | None = None

@dataclass(frozen=True)
class TrainingConfig:
    model: ModelConfig
    fine_tune: FineTuneConfig = field(default_factory=FineTuneConfig)

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
    # Random pad-and-crop: pad each side by N pixels then crop a random target_size
    # window. 0 disables. Standard CIFAR recipe uses 4.
    random_crop_padding: int = 0
    # Photometric jitters applied per image. Each value V means a multiplicative
    # factor sampled uniformly from [max(0, 1 - V), 1 + V]. 0.0 disables.
    # Saturation jitter is ignored for grayscale images.
    brightness_jitter: float = 0.0
    contrast_jitter: float = 0.0
    saturation_jitter: float = 0.0

# ---------------------------------------------------------------------------
# ISP pipeline config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ISPBlackLevelConfig:
    enabled: bool = True
    # Per-channel [R, G1, G2, B]; None = read from DNG metadata
    black_level: tuple[float, ...] | None = None

@dataclass(frozen=True)
class ISPDemosaicingConfig:
    algorithm: str = "bilinear"  # bilinear | malvar2004 | menon2007

@dataclass(frozen=True)
class ISPWhiteBalanceConfig:
    # Per-channel gains (g_gain normalized to 1.0); None = read from DNG metadata
    r_gain: float | None = None
    g_gain: float | None = None
    b_gain: float | None = None

@dataclass(frozen=True)
class ISPColorCorrectionConfig:
    enabled: bool = True
    # 9 floats (3×3, row-major); None = read from DNG metadata
    matrix: tuple[float, ...] | None = None

@dataclass(frozen=True)
class ISPDenoisingConfig:
    algorithm: str = "gaussian"  # gaussian | median | none
    strength: float = 0.5        # sigma for gaussian; ×10 rounded to odd int for median

@dataclass(frozen=True)
class ISPSharpeningConfig:
    algorithm: str = "unsharp_mask"  # unsharp_mask | sharpen_filter | none
    radius: float = 1.0
    amount: float = 1.0

@dataclass(frozen=True)
class ISPGammaConfig:
    gamma: float = 2.2

@dataclass(frozen=True)
class ISPConfig:
    black_level_correction: ISPBlackLevelConfig = field(default_factory=ISPBlackLevelConfig)
    demosaicing: ISPDemosaicingConfig = field(default_factory=ISPDemosaicingConfig)
    white_balance: ISPWhiteBalanceConfig = field(default_factory=ISPWhiteBalanceConfig)
    color_correction: ISPColorCorrectionConfig = field(default_factory=ISPColorCorrectionConfig)
    denoising: ISPDenoisingConfig = field(default_factory=ISPDenoisingConfig)
    sharpening: ISPSharpeningConfig = field(default_factory=ISPSharpeningConfig)
    gamma_correction: ISPGammaConfig = field(default_factory=ISPGammaConfig)
    gpu_accelerated: bool = False


@dataclass(frozen=True)
class ImagePreprocessingConfig:
    target_size: tuple[int, int] = (64, 64)  # (height, width)
    color_mode: str = "rgb"                   # "rgb" or "grayscale"
    normalize: bool = True                     # scale pixels to [0,1]
    flatten: bool = False                      # legacy sklearn field — ignored for CNN pipelines
    augmentation: ImageAugmentationConfig = field(default_factory=ImageAugmentationConfig)
    raw_input: bool = False                    # if True, load DNG files and run ISP pipeline
    isp: ISPConfig | None = None               # required when raw_input=True

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

# Regression evaluation currently needs no configurable options: the regression
# metrics (MAE, MSE, RMSE, R²) take no parameters. This placeholder keeps the
# evaluation-config structure symmetric with ClassificationEvalConfig and gives
# future regression-specific options a place to live.
@dataclass(frozen=True)
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
    promotion_evaluation_split: str = "val"

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
class ReloadConfig:
    """Opt-in pipeline-side trigger for the API model-reload endpoint."""
    enabled: bool = False
    url: str = "http://localhost:8000/admin/reload"
    timeout_seconds: int = 5

@dataclass(frozen=True)
class DeploymentConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelServingConfig = field(default_factory=ModelServingConfig)
    healthcheck: HealthcheckConfig = field(default_factory=HealthcheckConfig)
    reload: ReloadConfig = field(default_factory=ReloadConfig)

# ---------------------------------------------------------------------------
# Drift config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DriftStatTestConfig:
    numerical: str = "ks"
    categorical: str = "chisquare"

@dataclass(frozen=True)
class DriftStatTestThresholdConfig:
    numerical: float = 0.05
    categorical: float = 0.05

@dataclass(frozen=True)
class DriftSeverityConfig:
    low_max: float = 0.25
    medium_max: float = 0.50

@dataclass(frozen=True)
class DriftFeatureSeverityConfig:
    high_below: float = 0.001
    medium_below: float = 0.01

@dataclass(frozen=True)
class DriftMonitoringConfig:
    enabled: bool = True
    min_batch_size: int = 10
    alert_severity: str = "medium"
    # Conservative default: only "high" severity trips the CI gate.
    # Set to "never" to disable the non-interactive exit-code gate entirely.
    fail_on_severity: str = "high"

@dataclass(frozen=True)
class DriftImageSeverityConfig:
    # Wasserstein distance thresholds (z-scored images)
    medium: float = 0.10
    high: float = 0.25

@dataclass(frozen=True)
class DriftImageConfig:
    severity: DriftImageSeverityConfig = field(default_factory=DriftImageSeverityConfig)

@dataclass(frozen=True)
class DriftConfig:
    enabled: bool = True
    reference_source: str = "train"
    stattest: DriftStatTestConfig = field(default_factory=DriftStatTestConfig)
    stattest_threshold: DriftStatTestThresholdConfig = field(
        default_factory=DriftStatTestThresholdConfig
    )
    drift_share: float = 0.5
    severity: DriftSeverityConfig = field(default_factory=DriftSeverityConfig)
    feature_severity: DriftFeatureSeverityConfig = field(
        default_factory=DriftFeatureSeverityConfig
    )
    monitoring: DriftMonitoringConfig = field(default_factory=DriftMonitoringConfig)
    image: DriftImageConfig = field(default_factory=DriftImageConfig)
