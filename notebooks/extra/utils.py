from pathlib import Path
from hydra import compose, initialize
from omegaconf import OmegaConf


def load_config():
    try:
        with initialize(version_base=None, config_path="../../conf"):
            cfg = compose(config_name="config")
            print("Config loaded successfully!\n")
            print(OmegaConf.to_yaml(cfg))

        return cfg

    except Exception as e:
        print("Failed to load config:", e)
        raise