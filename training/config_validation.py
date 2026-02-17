"""YAML config validation and type-safe loading for NST training.

Addresses the YAML `safe_load` footgun where scientific notation like `1e-3`
is parsed as the string ``'1e-3'`` rather than the float ``0.001``.  Every
numeric training hyperparameter is explicitly cast and range-checked here.

Usage:
    from training.config_validation import load_and_validate_config
    cfg = load_and_validate_config("configs/multi_digit_cegis.yaml")
"""

from __future__ import annotations

import logging
import math
from typing import Any

import yaml

logger = logging.getLogger("config_validation")

# ── Type-cast specification ──────────────────────────────────
# key → (target_type, default, lo, hi)
# ``None`` for lo/hi means no bound.

_NUMERIC_FIELDS: dict[str, tuple[type, Any, Any, Any]] = {
    # Training section
    "epochs":               (int,   30,   1,     10_000),
    "lr":                   (float, 1e-3, 1e-8,  10.0),
    "batch_size":           (int,   64,   1,     4096),
    "seed":                 (int,   42,   0,     2**31),
    "lambda_constraint":    (float, 0.5,  0.0,   1000.0),
    "grad_clip":            (float, 1.0,  0.0,   None),

    # Lagrangian section
    "epsilon":              (float, 0.05, 0.0,   10.0),
    "alpha":                (float, 0.01, 0.0,   10.0),
    "rho":                  (float, 1.0,  0.0,   100.0),
    "lam_max":              (float, 10.0, 0.0,   1e6),
    "lagrangian_epsilon":   (float, 0.05, 0.0,   10.0),
    "lagrangian_alpha":     (float, 0.01, 0.0,   10.0),
    "lagrangian_rho":       (float, 1.0,  0.0,   100.0),
    "lagrangian_lam_max":   (float, 10.0, 0.0,   1e6),

    # CEGIS section
    "max_rounds":           (int,   10,   1,     1000),
    "inner_epochs":         (int,   15,   1,     1000),
    "max_counterexamples":  (int,   500,  1,     100_000),
    "ce_oversample":        (int,   3,    1,     100),

    # Baseline section
    "rounds":               (int,   10,   1,     1000),
    "replay_size":          (int,   500,  1,     100_000),
    "mine_size":            (int,   500,  1,     100_000),
    "oversample":           (int,   3,    1,     100),
    "eval_every":           (int,   5,    1,     1000),
    "dev_subset_size":      (int,   500,  1,     100_000),
    "max_rounds_quick":     (int,   3,    1,     100),

    # Data section
    "n_train":              (int,   5000, 1,     1_000_000),
    "n_test":               (int,   2000, 1,     1_000_000),
    "n_verify":             (int,   2000, 1,     1_000_000),
    "img_size":             (int,   28,   8,     224),
    "max_train_depth":      (int,   3,    1,     20),
    "max_test_depth":       (int,   6,    1,     20),
    "max_seq_len":          (int,   384,  16,    4096),
    "n_distractors":        (int,   2,    0,     20),
    "corruption_rate":      (float, 0.0,  0.0,   1.0),

    # Model section
    "d_model":              (int,   128,  8,     4096),
    "n_heads":              (int,   4,    1,     64),
    "n_layers":             (int,   2,    1,     64),
    "d_ff":                 (int,   256,  8,     16384),
    "dropout":              (float, 0.1,  0.0,   1.0),
}


class ConfigValidationError(ValueError):
    """Raised when a config value fails validation."""
    pass


def cast_value(key: str, value: Any) -> Any:
    """Cast a single config value to its expected type.

    Handles the YAML ``1e-3`` → ``'1e-3'`` string pitfall by attempting
    ``float(value)`` before ``int(value)`` for numeric fields.

    Returns the cast value, or the original if the key is unknown.
    """
    if key not in _NUMERIC_FIELDS:
        return value

    target_type, default, lo, hi = _NUMERIC_FIELDS[key]

    if value is None:
        return default

    # Cast string → number
    try:
        if target_type is int:
            # Allow "1e2" → 100 as well
            value = int(float(value))
        else:
            value = float(value)
    except (ValueError, TypeError) as exc:
        raise ConfigValidationError(
            f"Config key '{key}': cannot cast {value!r} to {target_type.__name__}"
        ) from exc

    # Range check
    if lo is not None and value < lo:
        raise ConfigValidationError(
            f"Config key '{key}' = {value} is below minimum {lo}"
        )
    if hi is not None and value > hi:
        raise ConfigValidationError(
            f"Config key '{key}' = {value} is above maximum {hi}"
        )

    # NaN / Inf check
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ConfigValidationError(
            f"Config key '{key}' = {value} is NaN or Inf"
        )

    return value


def validate_section(section: dict, section_name: str = "") -> dict:
    """Cast and validate all known numeric fields in a config section.

    Unknown keys are left untouched.  Returns a *new* dict.
    """
    out: dict[str, Any] = {}
    prefix = f"[{section_name}] " if section_name else ""
    for key, value in section.items():
        if isinstance(value, dict):
            # Recurse for nested sections
            out[key] = validate_section(value, key)
        else:
            try:
                out[key] = cast_value(key, value)
            except ConfigValidationError as exc:
                raise ConfigValidationError(f"{prefix}{exc}") from exc
    return out


def load_and_validate_config(path: str) -> dict:
    """Load a YAML config file and validate/cast all numeric fields.

    Returns the validated config dict with proper Python types.
    Raises ``ConfigValidationError`` on invalid values.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ConfigValidationError(f"Config file is empty: {path}")
    if not isinstance(raw, dict):
        raise ConfigValidationError(f"Config file root must be a mapping: {path}")

    validated = validate_section(raw, section_name="root")

    logger.debug("Validated config from %s", path)
    return validated


def validate_training_numerics(cfg: dict) -> dict:
    """Quick convenience: validate just the most critical training fields.

    Call this on already-loaded cfg dict to ensure lr, epochs, etc. are
    proper numeric types.  Returns a *new* copy of the config.
    """
    return validate_section(cfg, section_name="runtime")
