import math
from pathlib import Path

import json
from typing import Any
import pandas as pd

def atomic_write_text(path: Path, text: str) -> None:
    """Write text to a file atomically (write to temp + rename).

    Args:
        path (Path): The path to the file to write.
        text (str): The text content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    
def atomic_write_json(path: Path, data: dict) -> None:
    """Write a dictionary to a JSON file atomically, ensuring the output is JSON-compliant.

    Args:
        path (Path): The path to the JSON file to write.
        data (dict): The dictionary to write to JSON. Will be sanitized to ensure JSON compliance.
    """
    safe = sanitize_for_json(data)
    text = json.dumps(safe, indent=2, allow_nan=False)
    atomic_write_text(path, text)

def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV atomically (write to temp + rename).

    Args:
        df (pd.DataFrame): The DataFrame to write to CSV.
        path (Path): The path to the CSV file to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    

    
def atomic_write_pickle(obj: Any, path: Path) -> None:
    """Serialize obj to path using joblib, atomically (write-to-temp + rename).

    Uses joblib.dump (the standard sklearn persistence backend).
    Follows the same atomic pattern as atomic_write_csv and atomic_write_json:
    write to a .tmp sibling, then replace the target in one OS operation to
    prevent partial files from being observed by concurrent readers.

    Security note: only load .pkl files from this project's own artefact store.
    Never load pickle files from untrusted external sources.

    Args:
        obj:  Any joblib-serializable object (e.g., a fitted sklearn Pipeline).
        path: Destination path for the serialized file.
    """
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp)
    tmp.replace(path)


def sanitize_for_json(obj: Any) -> Any:
    """
    Convert values that are not strict-JSON compliant into safe equivalents.
    - NaN/Inf -> None
    - pandas/numpy scalars -> python scalars
    - dict keys -> strings (JSON requires string keys)
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    try:
        import numpy as np 
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            f = float(obj)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
    except Exception:
        pass
    if isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    return str(obj)