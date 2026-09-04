from __future__ import annotations

import enum
import functools

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FailMode(enum.StrEnum):
    open = "open"
    closed = "closed"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT", ge=1, le=65535)

    # Upstream LLM (OpenAI-compatible)
    upstream_base_url: str = Field(default="https://api.openai.com/v1", alias="UPSTREAM_BASE_URL")
    upstream_api_key: str = Field(default="", alias="UPSTREAM_API_KEY")
    upstream_model: str = Field(default="gpt-4o-mini", alias="UPSTREAM_MODEL")
    upstream_timeout_seconds: float = Field(default=60.0, alias="UPSTREAM_TIMEOUT_SECONDS", gt=0)

    # Security / pipeline toggles
    fail_mode: FailMode = Field(default=FailMode.closed, alias="FAIL_MODE")
    enable_provenance: bool = Field(default=True, alias="ENABLE_PROVENANCE")
    enable_normalizer: bool = Field(default=True, alias="ENABLE_NORMALIZER")
    enable_heuristic: bool = Field(default=True, alias="ENABLE_HEURISTIC")
    enable_classifier: bool = Field(default=False, alias="ENABLE_CLASSIFIER")
    enable_output_guard: bool = Field(default=True, alias="ENABLE_OUTPUT_GUARD")

    # Heuristic
    heuristic_rules_path: str = Field(default="rules/heuristic.yaml", alias="HEURISTIC_RULES_PATH")
    heuristic_perplexity_enabled: bool = Field(default=False, alias="HEURISTIC_PERPLEXITY_ENABLED")
    heuristic_perplexity_model: str = Field(
        default="distilgpt2", alias="HEURISTIC_PERPLEXITY_MODEL"
    )

    # Classifier
    classifier_model: str = Field(
        default="protectai/deberta-v3-base-prompt-injection-v2", alias="CLASSIFIER_MODEL"
    )
    classifier_threshold: float = Field(default=0.75, alias="CLASSIFIER_THRESHOLD", ge=0, le=1)
    classifier_batch_size: int = Field(default=8, alias="CLASSIFIER_BATCH_SIZE", ge=1, le=256)
    classifier_max_length: int = Field(default=512, alias="CLASSIFIER_MAX_LENGTH", ge=8, le=8192)
    classifier_device: str = Field(default="cpu", alias="CLASSIFIER_DEVICE")
    classifier_onnx_path: str | None = Field(default=None, alias="CLASSIFIER_ONNX_PATH")

    # XAI / Explainability (Phase 8)
    xai_attention_enabled: bool = Field(default=True, alias="XAI_ATTENTION_ENABLED")
    xai_top_k: int = Field(default=10, alias="XAI_TOP_K", ge=1, le=100)
    xai_ig_enabled: bool = Field(default=False, alias="XAI_IG_ENABLED")
    xai_ig_steps: int = Field(default=32, alias="XAI_IG_STEPS", ge=2, le=512)
    xai_store_enabled: bool = Field(default=True, alias="XAI_STORE_ENABLED")

    # Multiturn
    multiturn_window: int = Field(default=6, alias="MULTITURN_WINDOW", ge=1, le=64)
    multiturn_cumulative_threshold: float = Field(
        default=1.2, alias="MULTITURN_CUMULATIVE_THRESHOLD", gt=0
    )

    # Decision thresholds (confidence 0..1)
    threshold_log: float = Field(default=0.3, alias="THRESHOLD_LOG", ge=0, le=1)
    threshold_sanitize: float = Field(default=0.55, alias="THRESHOLD_SANITIZE", ge=0, le=1)
    threshold_exclude: float = Field(default=0.7, alias="THRESHOLD_EXCLUDE", ge=0, le=1)
    threshold_block: float = Field(default=0.85, alias="THRESHOLD_BLOCK", ge=0, le=1)

    # DB / Redis
    database_url: str = Field(default="sqlite+aiosqlite:///./waf.db", alias="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # Observability
    otel_exporter_otlp_endpoint: str = Field(default="", alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    prometheus_enabled: bool = Field(default=True, alias="PROMETHEUS_ENABLED")

    # Canary
    canary_enabled: bool = Field(default=True, alias="CANARY_ENABLED")
    canary_length: int = Field(default=16, alias="CANARY_LENGTH", ge=8, le=128)

    # Request logging (dashboard data source)
    log_requests_enabled: bool = Field(default=True, alias="LOG_REQUESTS_ENABLED")

    # Alerts (Phase 9.3)
    alerts_enabled: bool = Field(default=False, alias="ALERTS_ENABLED")
    alerts_webhook_url: str = Field(default="", alias="ALERTS_WEBHOOK_URL")
    alerts_min_level: str = Field(default="block", alias="ALERTS_MIN_LEVEL")
    alerts_timeout_seconds: float = Field(default=5.0, alias="ALERTS_TIMEOUT_SECONDS", gt=0, le=60)

    # Continuous retraining (Phase 10)
    retrain_enabled: bool = Field(default=False, alias="RETRAIN_ENABLED")
    # confidence floor for auto-collecting borderline/blocked requests
    retrain_borderline_min: float = Field(default=0.3, alias="RETRAIN_BORDERLINE_MIN", ge=0, le=1)
    retrain_max_queue: int = Field(default=10000, alias="RETRAIN_MAX_QUEUE", ge=1)
    retrain_corpus_dir: str = Field(default="data/corpora", alias="RETRAIN_CORPUS_DIR")
    retrain_dataset_dir: str = Field(default="data/datasets", alias="RETRAIN_DATASET_DIR")
    retrain_val_split: float = Field(default=0.1, alias="RETRAIN_VAL_SPLIT", ge=0, le=0.5)
    retrain_seed: int = Field(default=42, alias="RETRAIN_SEED", ge=0)
    retrain_registry_path: str = Field(
        default="models/registry.json", alias="RETRAIN_REGISTRY_PATH"
    )

    @model_validator(mode="after")
    def validate_threshold_order(self) -> Settings:
        thresholds = (
            self.threshold_log,
            self.threshold_sanitize,
            self.threshold_exclude,
            self.threshold_block,
        )
        if list(thresholds) != sorted(thresholds):
            raise ValueError("thresholds must be ordered: log <= sanitize <= exclude <= block")
        return self

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() in ("development", "dev", "local")


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
