import os
from pathlib import Path

def load_config(env_path: str = None) -> dict:
    """Loads environment variables from config.env into os.environ and returns a dict."""
    if env_path is None:
        env_path = Path(__file__).parent / "config.env"
    else:
        env_path = Path(env_path)

    config_vars = {}
    if not env_path.exists():
        print(f"Warning: {env_path} not found. Using current environment variables.")
        return dict(os.environ)

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                os.environ[key] = val
                config_vars[key] = val
    return config_vars

def print_config():
    """Prints active JAX GKE configuration variables."""
    cfg = load_config()
    print("=" * 50)
    print("  JAX Multi-Node GKE Active Configuration")
    print("=" * 50)
    for k, v in cfg.items():
        if k in ["PROJECT_ID", "REGION", "ZONE", "CLUSTER_NAME", "GPU_TYPE", "TPU_TYPE", "ARTIFACT_REGISTRY_REPO"]:
            print(f"  {k:25s}: {v}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
