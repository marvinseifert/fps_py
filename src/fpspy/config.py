"""
Allow loading and saving of options from a TOML config file, with default fallbacks.
"""
import importlib.resources
import os
from pathlib import Path
import platformdirs
import logging
import datetime
from typing import Optional
import copy

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
    path = Path(platformdirs.user_config_dir(appname=APP_NAME, appauthor=APP_AUTHOR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_cache_dir() -> Path:
    """Get the user cache directory for fpspy."""
    path = Path(platformdirs.user_cache_dir(appname=APP_NAME, appauthor=APP_AUTHOR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_data_dir() -> Path:
    """Get the user data directory for fpspy."""
    path = Path(platformdirs.user_data_dir(appname=APP_NAME, appauthor=APP_AUTHOR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_data_dir(config: dict) -> Path:
    """Return the effective data directory."""
    # 1. If config has paths.data_dir and it's non-empty, use that
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


def user_log_dir(config: dict) -> Path:
    """Return the effective log directory."""
    # 1. If config has paths.log_dir and it's non-empty, use that
    paths_cfg = config.get("paths", {})
    log_dir_str = paths_cfg.get("log_dir", "").strip()
    if log_dir_str:
        path = Path(log_dir_str).expanduser()
    else:
        # 2. Fall back to default platformdirs location
        path = default_log_dir()

    path.mkdir(parents=False, exist_ok=True)
    return path


def default_log_dir() -> Path:
    """Get the user logs directory for fpspy.

    Actual runs will create a timestamped subdirectory inside this (unless a custom
    output directory is used).
    """
    path = Path(platformdirs.user_log_dir(appname=APP_NAME, appauthor=APP_AUTHOR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timestamp_dirname():
    """Get a timestamped subdirectory name for logs."""
    now = datetime.datetime.now(datetime.timezone.utc)
    res = now.strftime("%Y%m%dT%H%M%SZ")
    return res


def create_outdir(config) -> Path:
    """Create a timestamped output directory for logs."""
    out_dir = user_log_dir(config) / _timestamp_dirname()
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _load_default_config() -> dict:
    """Load the default config file from the package resources."""
    resource_dir = importlib.resources.files("fpspy.resources")
    cfg_path = Path(resource_dir / "default_settings.toml")
    if not cfg_path.exists():
        raise FileNotFoundError("Default config file not found in package resources.")
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
    log_dir_str = paths.get("log_dir", "").strip()

    if data_dir_str:
        data_dir = Path(os.path.expandvars(data_dir_str)).expanduser()
    else:
        # fallback to system user data dir
        data_dir = default_data_dir()
    if log_dir_str:
        log_dir = Path(os.path.expandvars(log_dir_str)).expanduser()
    else:
        # fallback to system user log dir
        log_dir = default_log_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    paths["data_dir"] = str(data_dir)
    paths["log_dir"] = str(log_dir)
    return config

def _expand_path_str(value: str, mapping: dict[str, str]) -> str:
      # $HOME, %USERPROFILE%
      value = os.path.expandvars(value)
      # ~
      value = os.path.expanduser(value)          
      for token, replacement in mapping.items():
          value = value.replace("{" + token + "}", replacement)
      return value

def _expand_config_paths(config: dict) -> dict:
    mapping = {
            "data_dir": config["paths"]["data_dir"],
            "log_dir": config["paths"]["log_dir"],
            "config_dir": str(user_config_dir()),
    }
    def walk(d):
        for key, val in d.items():
            # Don't expand [paths] section.
            skip = key == "paths"
            if not skip and isinstance(val, dict):
                walk(val)
            elif isinstance(val, str):
                if key.endswith("_path") or key.endswith("_dir"):
                    d[key] = _expand_path_str(val, mapping)
        return d
    return walk(config)


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


def _apply_window_defaults(config: dict) -> dict:
    """Merge `windows.default` into each numbered window definition."""
    windows = config.get("windows", {})
    template = windows.pop("default", None)
    if template is None:
        return config
    for key, win in list(windows.items()):
        if isinstance(win, dict):
            merged = copy.deepcopy(template)
            _deep_merge(merged, win)
            windows[key] = merged
    return config


def load_config(path: Optional[Path] = None, overrides: Optional[dict] = None) -> dict:
    """
    Load config with default-fallback behavior.

    Priority, lowest to highest:
    1. default_settings.toml (base)
    2. merged with:
       a) explicit path passed by CLI, or
       b) user config file in standard location, if present
    3. overrides passed to function
    4. Finally, and this might seem unusual, but merge window defaults at the end. This
        is done at the end, as we don't yet know how many windows there will be.
    """
    # 1. Load default config
    default_cfg = _load_default_config()

    # 2. Load user config or explicit path
    if path is not None:
        config = _load_config_from_path(path)
        _logger.info(f"{path} [config file]")
        config = _deep_merge(default_cfg, config)
    else:
        config = _load_user_config()
        if config is None:
            _logger.info("Loaded bundled default config (no user config).")
            config = default_cfg
        else:
            _logger.info(f"Loaded user config: {user_config_file_path()}")
            config = _deep_merge(default_cfg, config)

    # 3. Apply overrides passed to function
    if overrides is not None:
        config = _deep_merge(config, overrides)

    # 4.
    config = _apply_window_defaults(config)
    config = _resolve_paths(config)
    config = _expand_config_paths(config)
    _logger.info(f"Effective config (toml):\n{config_to_str(config)}")
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


"""
Simple accessors for common config values. Saves having hardcoded strings
"""


def get_arduino_port(config: dict) -> str:
    """Get the Arduino port from config."""
    return config["arduino"]["port"]


def get_arduino_baud_rate(config: dict) -> int:
    """Get the Arduino baud rate from config."""
    return config["arduino"]["baud_rate"]


def get_arduino_trigger_command(config: dict) -> str:
    """Get the Arduino trigger command from config."""
    return config["arduino"]["trigger_command"]


def get_presentation_delay(config: dict) -> float:
    """Get the global delay before stimulus start from config."""
    return config["presentation_delay"]
