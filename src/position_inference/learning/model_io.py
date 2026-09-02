from pathlib import Path
from typing import Optional
import joblib

from position_inference.learning.role_model import ViewSpecificRoleModel


def save_role_model(model: ViewSpecificRoleModel, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_role_model(path: Path) -> Optional[ViewSpecificRoleModel]:
    path = Path(path)
    if not path.exists():
        return None
    return joblib.load(path)
