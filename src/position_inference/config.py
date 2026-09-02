from pathlib import Path
from typing import Any, Dict
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def load_yaml_config(filename: str) -> Dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_position_taxonomy() -> Dict[str, Any]:
    return load_yaml_config("position_taxonomy.yaml")


def get_action_role_rules() -> Dict[str, Any]:
    return load_yaml_config("action_role_rules.yaml")


def get_scoring_weights() -> Dict[str, Any]:
    return load_yaml_config("scoring_weights.yaml")


def get_pairing_config() -> Dict[str, Any]:
    return load_yaml_config("pairing.yaml")


def get_confidence_config() -> Dict[str, Any]:
    return load_yaml_config("confidence.yaml")


def get_personnel_constraints() -> Dict[str, Any]:
    return load_yaml_config("personnel_constraints.yaml")

