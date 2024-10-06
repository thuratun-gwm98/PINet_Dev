import os
from typing import Dict
import yaml

config_path = os.path.join(os.path.dirname(__file__), "configs.yaml")
def __get_config(part: str) -> Dict:
    with open(config_path) as cfg_file:
        config: Dict = yaml.load(cfg_file, yaml.SafeLoader)
        return config[part]

# fmt: off
DATASET_CFG = __get_config("dataset")
OPTIMIZER_CFG = __get_config("optimizer")
TRAINER_CFG = __get_config("trainer")
LOSS_CFG = __get_config("loss")
# MODEL_CFG = __get_config("model")
# fmt: on

