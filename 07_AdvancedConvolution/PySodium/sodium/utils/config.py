import yaml


def load_config(filename: str) -> dict:
    """
    Load a configuration file as YAML
    """
    with open(filename) as fh:
        config = yaml.safe_load(fh)

    return config
