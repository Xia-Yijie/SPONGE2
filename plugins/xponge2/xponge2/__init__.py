"""xponge2 — Python plugin for SPONGE input generation."""

from pathlib import Path

from .assign import assignment_to_residue_type
from .config import config_path
from .config import get_config
from .config import load_config
from .config import project_config_path
from .config import reset_config
from .config import save_config
from .config import set_config
from .core import Molecule
from .core import ResidueType
from .forcefield.amber.registry import register_amber_forcefield_data
from .io.sponge import save_sponge_input

__version__ = "2.0.0-beta.1"

try:
    from . import _core
    from ._core import Assign
    from ._core import GaffParameters
    from ._core import Get_Assignment_From_Mol2
    from ._core import generate_gaff_frcmod as _native_generate_gaff_frcmod
    from ._core import get_assignment_from_mol2
    from ._core import load_frcmod as _native_load_frcmod
    from ._core import load_gaff_parameters as _native_load_gaff_parameters
except ImportError as exc:
    _CORE_IMPORT_ERROR = exc
    _core = None

    def _core_unavailable(*args, **kwargs):
        raise ImportError(
            "xponge2._core extension is not available"
        ) from _CORE_IMPORT_ERROR

    class Assign:
        __init__ = _core_unavailable

    get_assignment_from_mol2 = _core_unavailable
    Get_Assignment_From_Mol2 = _core_unavailable
    _native_generate_gaff_frcmod = _core_unavailable
    GaffParameters = _core_unavailable
    _native_load_frcmod = _core_unavailable
    _native_load_gaff_parameters = _core_unavailable


def generate_gaff_frcmod(
    ifname,
    ofname,
    *,
    ffset=1,
    print_all=False,
    print_dihedral_contain_X=True,
    datapath=None,
):
    if datapath is None:
        datapath = get_config("parmchk2_data_dir")
    _native_generate_gaff_frcmod(
        str(ifname),
        str(ofname),
        ffset=ffset,
        print_all=print_all,
        print_dihedral_contain_X=print_dihedral_contain_X,
        datapath=str(datapath),
    )


def load_gaff_parameters(dat_path=None, frcmod_path=None):
    if dat_path is None:
        dat_path = get_config("gaff_dat_path")
    if frcmod_path is not None:
        frcmod_path = str(frcmod_path)
    return _native_load_gaff_parameters(str(dat_path), frcmod_path=frcmod_path)


def load_frcmod(filename):
    return _native_load_frcmod(str(filename))


Assign.to_residuetype = assignment_to_residue_type


__all__ = [
    "Assign",
    "GaffParameters",
    "Get_Assignment_From_Mol2",
    "Molecule",
    "ResidueType",
    "assignment_to_residue_type",
    "config_path",
    "generate_gaff_frcmod",
    "get_assignment_from_mol2",
    "get_config",
    "load_config",
    "load_frcmod",
    "load_gaff_parameters",
    "project_config_path",
    "register_amber_forcefield_data",
    "reset_config",
    "save_config",
    "save_sponge_input",
    "set_config",
]
