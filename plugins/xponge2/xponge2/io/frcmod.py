import importlib

from ..config import get_config


_core = importlib.import_module("xponge2._core")
GaffParameters = _core.GaffParameters


def load_gaff_parameters(dat_path=None, frcmod_path=None):
    if dat_path is None:
        dat_path = get_config("gaff_dat_path")
    if frcmod_path is not None:
        frcmod_path = str(frcmod_path)
    return _core.load_gaff_parameters(str(dat_path), frcmod_path=frcmod_path)


def load_frcmod(filename):
    return _core.load_frcmod(str(filename))
