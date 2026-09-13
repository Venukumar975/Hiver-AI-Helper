import os
import configparser
from typing import List, Dict, Any, Optional

def load_models_from_ini(ini_path: str = "models.ini") -> List[str]:
    """
    Reads models.ini and returns the list of model names ordered by priority (priority_1, priority_2, ...).
    """
    if not os.path.exists(ini_path):
        # Fallback default list if models.ini is not found
        return [
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3-flash-preview",
            "gemini-flash-lite-latest"
        ]

    config = configparser.ConfigParser()
    config.read(ini_path, encoding="utf-8")

    if not config.has_section("gemini_models"):
        return []

    # Extract all priority_X keys and sort by integer X
    priority_items = []
    for key, val in config.items("gemini_models"):
        key_clean = key.lower().strip()
        if key_clean.startswith("priority_"):
            try:
                num = int(key_clean.split("_")[1])
                priority_items.append((num, val.strip()))
            except ValueError:
                continue

    # Sort by priority number (1, 2, 3...)
    priority_items.sort(key=lambda x: x[0])
    return [item[1] for item in priority_items]
