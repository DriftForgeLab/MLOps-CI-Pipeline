# =============================================================================
# src/config/loader.py — Public facade for configuration loading
# =============================================================================
# Re-exports all load_* functions and dataclasses so that existing imports
# like `from src.config.loader import load_config, PipelineConfig` continue
# to work without any changes.
#
# The actual loading logic lives in domain-specific modules:
#   pipeline_loader.py      — load_config / PipelineConfig
#   training_loader.py      — load_training_config / TrainingConfig
#   preprocessing_loader.py — load_preprocessing_config / PreprocessingConfig
#   evaluation_loader.py    — load_evaluation_config / EvaluationConfig
#   promotion_loader.py     — load_promotion_config / PromotionConfig
#   deployment_loader.py    — load_deployment_config / DeploymentConfig
# =============================================================================

from src.config.pipeline_loader import load_config  # noqa: F401
from src.config.training_loader import load_training_config  # noqa: F401
from src.config.preprocessing_loader import load_preprocessing_config  # noqa: F401
from src.config.evaluation_loader import load_evaluation_config  # noqa: F401
from src.config.promotion_loader import load_promotion_config  # noqa: F401
from src.config.deployment_loader import load_deployment_config  # noqa: F401
from src.config.drift_loader import load_drift_config  # noqa: F401

# Re-export dataclasses so `from src.config.loader import PipelineConfig` works.
from src.config.schema import (  # noqa: F401
    ProjectMeta,
    DataPaths,
    SubConfigPaths,
    MLflowConfig,
    PipelineConfig,
    RandomForestHyperparams,
    LogisticRegressionHyperparams,
    LinearRegressionHyperparams,
    CnnConvLayerConfig,
    CnnArchitectureConfig,
    CnnHyperparams,
    ModelConfig,
    TrainingConfig,
    EncodingConfig,
    ScalingConfig,
    MissingValuesConfig,
    ImageAugmentationConfig,
    ISPBlackLevelConfig,
    ISPDemosaicingConfig,
    ISPWhiteBalanceConfig,
    ISPColorCorrectionConfig,
    ISPDenoisingConfig,
    ISPSharpeningConfig,
    ISPGammaConfig,
    ISPConfig,
    ImagePreprocessingConfig,
    PreprocessingConfig,
    ClassificationEvalConfig,
    RegressionEvalConfig,
    EvaluationConfig,
    PromotionRule,
    PromotionTaskConfig,
    PromotionConfig,
    ServerConfig,
    ModelServingConfig,
    HealthcheckConfig,
    DeploymentConfig,
    DriftStatTestConfig,
    DriftStatTestThresholdConfig,
    DriftSeverityConfig,
    DriftFeatureSeverityConfig,
    DriftRecommendationConfig,
    DriftPipelineConfig,
    DriftMonitoringConfig,
    DriftConfig,
)
