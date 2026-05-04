from pathlib import Path

import yaml


def find_config_path(config_dir, model_version):
    config_dir = Path(config_dir)
    matches = sorted(config_dir.glob(f"*{model_version}*.y*ml"))

    if not matches:
        raise FileNotFoundError(
            f"No config file in {config_dir} matches model_version='{model_version}'"
        )

    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise RuntimeError(
            f"Multiple config files match model_version='{model_version}': {names}"
        )

    return matches[0]


def load_config(config_path):
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
