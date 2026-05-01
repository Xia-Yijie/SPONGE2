"""Runtime configuration for xponge2."""

from pathlib import Path

import tomlkit

from .forcefield.amber.parmchk2 import bundled_data_dir


USER_CONFIG_PATH = Path.home() / ".xponge" / "config.toml"
PROJECT_CONFIG_NAME = "xponge.toml"


def _default_settings():
    return {
        "parmchk2_data_dir": None,
        "gaff_dat_path": None,
        "gaff2_dat_path": None,
    }


_SETTINGS = _default_settings()


def config_path():
    return USER_CONFIG_PATH


def project_config_path(cwd=None):
    return Path(cwd or Path.cwd()) / PROJECT_CONFIG_NAME


def _apply_config_file(path):
    target = Path(path)
    if not target.exists():
        return
    data = tomlkit.parse(target.read_text())
    for key, value in data.items():
        if key in _SETTINGS:
            _SETTINGS[key] = None if value is None else str(value)


def _effective_settings():
    settings = dict(_SETTINGS)
    if settings["parmchk2_data_dir"] is None:
        settings["parmchk2_data_dir"] = str(bundled_data_dir())
    if settings["gaff_dat_path"] is None:
        settings["gaff_dat_path"] = str(Path(settings["parmchk2_data_dir"]) / "gaff.dat")
    if settings["gaff2_dat_path"] is None:
        settings["gaff2_dat_path"] = str(Path(settings["parmchk2_data_dir"]) / "gaff2.dat")
    return settings


def load_config(path=None, *, include_project=True, cwd=None):
    _SETTINGS.clear()
    _SETTINGS.update(_default_settings())
    if path is None:
        _apply_config_file(USER_CONFIG_PATH)
        if include_project:
            _apply_config_file(project_config_path(cwd))
    else:
        _apply_config_file(path)
    return _effective_settings()


def save_config(path=None):
    target = Path(path) if path is not None else USER_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    doc = tomlkit.document()
    for key, value in _SETTINGS.items():
        if value is not None:
            doc[key] = str(value)
    target.write_text(tomlkit.dumps(doc))
    return target


def get_config(key=None):
    settings = _effective_settings()
    if key is None:
        return settings
    return settings[str(key)]


def set_config(key, value, *, save=False):
    key = str(key)
    if key not in _SETTINGS:
        raise KeyError(f"unknown xponge2 config key: {key}")
    _SETTINGS[key] = None if value is None else str(value)
    if save:
        save_config()


def reset_config(*, save=False):
    _SETTINGS.clear()
    _SETTINGS.update(_default_settings())
    if save:
        save_config()
    return _effective_settings()


load_config()
