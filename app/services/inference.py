"""Load only the approved AI implementation; no model fallback or training."""

import hashlib
import importlib.util
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.errors import ServiceError

InferenceState = tuple[Any, dict, Callable]


@dataclass(frozen=True)
class ModelBundle:
    state: InferenceState
    checkpoint_digest: str
    policy_digest: str

    @property
    def model_version(self) -> str:
        return f"lstm_full_pw10:{self.checkpoint_digest}"

    @property
    def threshold_version(self) -> str:
        return f"locked-uncalibrated-0.2:{self.policy_digest}"

    def predict(self, observations: list[dict]) -> dict:
        model, stats, predict = self.state
        return predict(observations, model, stats)


def load_bundle(root: Path | None) -> ModelBundle:
    if root is None:
        raise RuntimeError("SEWS_AI_REPO_ROOT is required")
    module_path = root / "src/inference/predict_lstm.py"
    spec = importlib.util.spec_from_file_location(
        "sews_frozen_predict_lstm", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("AI inference module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model, stats = module.load_frozen_model(root=root)
    checkpoint_digest = module.file_sha256(root / module.MODEL_FILE)
    policy = json.loads((root / module.POLICY_FILE).read_text(encoding="utf-8"))
    # Version the full locked policy, not just the display threshold.
    from app.store.sqlite import canonical

    policy_digest = hashlib.sha256(canonical(policy).encode()).hexdigest()
    return ModelBundle((model, stats, module.predict), checkpoint_digest, policy_digest)


def predict_state(state: InferenceState | None, observations: list[dict]) -> dict:
    if state is None:
        raise ServiceError(503, "model_not_ready", "Model is not ready")
    model, stats, predict = state
    try:
        return predict(observations, model, stats)
    except (ValueError, TypeError):
        raise ServiceError(
            422, "invalid_observations", "Observations violate the model input contract"
        ) from None
