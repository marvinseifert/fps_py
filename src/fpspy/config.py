import importlib.resources
from pathlib import Path
import platformdirs
import logging

try:
    # Python 3.11+
    import tomllib
except ModuleNotFoundError:
    # Older Python versions
    import tomli as tomllib
import tomli_w

_logger = logging.getLogger(__name__)


APP_NAME = "fpspy"
APP_AUTHOR = "Marvin Seifert"
CONFIG_FILE_NAME = "settings.toml"


def user_config_dir() -> Path:
    """Get the user config directory for fpspy."""
    path = Path(
        platformdirs.user_config_dir(appname=APP_NAME, appauthor=APP_AUTHOR)
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_cache_dir() -> Path:
    """Get the user cache directory for fpspy."""
    path = Path(
        platformdirs.user_cache_dir(appname=APP_NAME, appauthor=APP_AUTHOR)
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_data_dir() -> Path:
    """Get the user data directory for fpspy."""
    path = Path(
        platformdirs.user_data_dir(appname=APP_NAME, appauthor=APP_AUTHOR)
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_data_dir(config: dict) -> Path:
    """Return the effective data directory."""
    # 1. If config has paths.data_root and it's non-empty, use that
    paths_cfg = config.get("paths", {})
    data_dir_str = paths_cfg.get("data_dir", "").strip()

    if data_dir_str:
        path = Path(data_dir_str).expanduser()
    else:
        # 2. Fall back to default platformdirs location
        path = default_data_dir()

    path.mkdir(parents=False, exist_ok=True)
    return path

def user_config_file_path() -> Path:
    """Get the path to the user config file."""
    return user_config_dir() / CONFIG_FILE_NAME


def user_log_dir() -> Path:
    """Get the user logs directory for fpspy."""
    path = Path(
        platformdirs.user_log_dir(appname=APP_NAME, appauthor=APP_AUTHOR)
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_arduino_port(config: dict) -> str:
    """Get the Arduino port from config."""
    return config["arduino"]["port"]


def get_arduino_baud_rate(config: dict) -> int:
    """Get the Arduino baud rate from config."""
    return config["arduino"]["baud_rate"]


def get_arduino_trigger_command(config: dict) -> str:
    """Get the Arduino trigger command from config."""
    return config["arduino"]["trigger_command"]


def _load_default_config() -> dict:
    """Load the default config file from the package resources."""
    resource_dir = importlib.resources.files("fpspy.resources")
    cfg_path = Path(resource_dir / "default_settings.toml")
    if not cfg_path.exists():
        raise FileNotFoundError(
            "Default config file not found in package resources."
        )
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def _load_config_from_path(path: Path) -> dict:
    """Load a user specified config file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as f:
        return tomllib.load(f)


def _load_user_config() -> dict | None:
    """Load the user config file from the user config directory."""
    config_path = user_config_file_path()
    if not config_path.exists():
        return None
    with config_path.open("rb") as f:
        return tomllib.load(f)


def _resolve_paths(config: dict) -> dict:
    """Resolve any paths not specified in the config."""
    paths = config.setdefault("paths", {})
    data_dir_str = paths.get("data_dir", "").strip()

    if data_dir_str:
        data_dir = Path(data_dir_str).expanduser()
    else:
        # fallback to system user data dir
        data_dir = default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    paths["data_dir"] = str(data_dir)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge `override` into `base`.

    - If a key exists in both and both values are dicts: merge recursively.
    - Otherwise: value from `override` wins.
    """
    for key, override_val in override.items():
        base_val = base.get(key)
        if isinstance(base_val, dict) and isinstance(override_val, dict):
            _deep_merge(base_val, override_val)
        else:
            base[key] = override_val
    return base


def config_to_str(config: dict) -> str:
    """Convert config dict to a TOML string."""
    res = tomli_w.dumps(config)
    return res


def load_config(path: Path | None = None) -> dict:
    """
    Load config with default-fallback behavior.

    Priority:
    1. default_settings.toml (base)
    2. merged with:
       a) explicit path passed by CLI, or
       b) user config file in standard location, if present
    """
    default_cfg = _load_default_config()

    if path is not None:
        config = _load_config_from_path(path)
        _logger.info(f"Loaded config from explicit path: {path}")
        config = _deep_merge(default_cfg, config)
    else:
        config = _load_user_config()
        if config is None:
            _logger.info("Loaded bundled default config (no user config).")
            config = default_cfg
        else:
            _logger.info(f"Loaded user config: {user_config_file_path()}")
            config = _deep_merge(default_cfg, config)

    config = _resolve_paths(config)
    _logger.info(f"Effective config:\n{config_to_str(config)}")
    return config


def save_user_config(config: dict) -> None:
    """
    Save the given config dict as TOML to the user config file.

    This will override <settings-dir>settings.toml, and create it if it does not exist
    yet.
    """
    config_path = user_config_file_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("wb") as f:
        tomli_w.dump(config, f)

    _logger.info(f"Saved user config to {config_path}")
