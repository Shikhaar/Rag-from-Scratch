"""
Persistence utilities for serialization and atomic disk operations.
"""

import json
import os
import shutil
import tempfile
from typing import Any, Dict


def atomic_write_json(filepath: str, data: Dict[str, Any], indent: int = 2) -> None:
    """Safely write JSON data to disk using an atomic rename to prevent corruption."""
    dir_name = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(dir_name, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
        json.dump(data, tf, indent=indent)
        temp_name = tf.name

    shutil.move(temp_name, filepath)


def load_json(filepath: str, default: Any = None) -> Any:
    """Load JSON data from disk if it exists, otherwise return default."""
    if not os.path.exists(filepath):
        return default
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
