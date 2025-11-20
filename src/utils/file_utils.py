import json
import yaml
from pathlib import Path


def read_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_yaml(path: str, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, data):
    with open(path, "w") as f:
        return json.dump(data, f, indent=4)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

