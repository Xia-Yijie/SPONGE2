from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._core import Assign, GaffParameters
from ._core import Get_Assignment_From_Mol2, get_assignment_from_mol2
from .assign import assignment_to_residue_type
from .config import config_path, get_config, load_config, project_config_path
from .config import reset_config, save_config, set_config
from .core import Molecule, ResidueType
from .forcefield.amber.registry import register_amber_forcefield_data
from .io.sponge import save_sponge_input

__version__: str

_FrcmodSections = tuple[str, str, str, str, str, str, dict[str, Any]]

def generate_gaff_frcmod(
    ifname: str | Path,
    ofname: str | Path,
    *,
    ffset: int = 1,
    print_all: bool = False,
    print_dihedral_contain_X: bool = True,
    datapath: str | Path | None = None,
) -> None: ...
def load_gaff_parameters(
    dat_path: str | Path | None = None, frcmod_path: str | Path | None = None
) -> GaffParameters: ...
def load_frcmod(filename: str | Path) -> _FrcmodSections: ...

__all__: list[str]
