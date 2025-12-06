import os
import json
from datetime import datetime
from typing import Dict

from src import config


def save_metrics(metrics: Dict, name: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config.METRICS_DIR, f"{name}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path
