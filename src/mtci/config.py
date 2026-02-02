from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.functional_validators import field_validator


class ConfigError(Exception):
    pass


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Tolerance(StrictBaseModel):
    atol: float = 0.0
    rtol: float = 0.01


class Profile(StrictBaseModel):
    budget_seconds: float = Field(..., gt=0)
    max_examples: int = Field(20, gt=0)
    retries_on_fail: int = Field(1, ge=0)
    fail_on_flake: bool = True
    tolerance: Tolerance = Tolerance()
    mrs: List[str]
    junit_flaky_as_failure: bool = True


class DatasetConfig(StrictBaseModel):
    path: str
    jsonl_field: str


class LocalModelConfig(StrictBaseModel):
    mode: Literal["local"]
    entrypoint: str
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class EndpointModelConfig(StrictBaseModel):
    mode: Literal["endpoint"]
    base_url: str
    predict_path: str = "/predict"
    timeout_s: float = 10.0


ModelConfig = LocalModelConfig | EndpointModelConfig


class Config(StrictBaseModel):
    profiles: Dict[str, Profile]
    dataset: DatasetConfig
    model: ModelConfig = Field(discriminator="mode")

    @field_validator("profiles")
    @classmethod
    def ensure_profiles(cls, value: Dict[str, Profile]) -> Dict[str, Profile]:
        if not value:
            raise ValueError("profiles must define at least one profile")
        return value


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML: {exc}") from exc
    if data is None:
        raise ConfigError("Config is empty")
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc)) from exc


def format_validation_error(exc: ValidationError) -> str:
    lines = ["Config validation failed:"]
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", []))
        msg = err.get("msg", "invalid")
        lines.append(f"- {loc}: {msg}")
    return "\n".join(lines)


@dataclass
class ResolvedProfile:
    name: str
    profile: Profile
