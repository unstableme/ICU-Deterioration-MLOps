import yaml
from pathlib import Path

PARAMS_PATH = Path(__file__).parent[1] / 'params.yaml'
def load_params():
    """Load parameters from params.yaml file."""
    with open(PARAMS_PATH, 'r') as file:
        return yaml.safe_load(file)