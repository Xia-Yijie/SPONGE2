from collections.abc import Mapping, Sequence
from typing import Any

from .core import ResidueType

_Coordinate = tuple[float, float, float]
_FrcmodSections = tuple[str, str, str, str, str, str, dict[str, Any]]


class Assign:
    def __init__(self, name: str = "ASN") -> None: ...

    name: str
    atom_numbers: int
    atoms: list[str]
    names: list[str]
    element_details: list[str]
    coordinate: list[_Coordinate]
    charge: list[float]
    formal_charge: list[int]
    bonds: dict[int, dict[int, int]]
    atom_marker: dict[int, dict[str, int]]
    bond_marker: dict[int, dict[int, set[str]]]
    atom_types: dict[int, str | None]
    built: bool
    kekulized: bool

    def add_atom(
        self,
        element: str,
        x: float,
        y: float,
        z: float,
        name: str = "",
        charge: float = 0.0,
    ) -> None: ...
    def Add_Atom(
        self,
        element: str,
        x: float,
        y: float,
        z: float,
        name: str = "",
        charge: float = 0.0,
    ) -> None: ...
    def add_bond(self, atom1: int, atom2: int, order: int = -1) -> None: ...
    def Add_Bond(self, atom1: int, atom2: int, order: int = -1) -> None: ...
    def add_atom_marker(self, atom: int, marker: str) -> None: ...
    def Add_Atom_Marker(self, atom: int, marker: str) -> None: ...
    def add_bond_marker(
        self, atom1: int, atom2: int, marker: str, only1: bool = False
    ) -> None: ...
    def Add_Bond_Marker(
        self, atom1: int, atom2: int, marker: str, only1: bool = False
    ) -> None: ...
    def delete_bond(self, atom1: int, atom2: int) -> None: ...
    def Delete_Bond(self, atom1: int, atom2: int) -> None: ...
    def check_connectivity(self) -> bool: ...
    def Check_Connectivity(self) -> bool: ...
    def determine_atom_type(self, rule: str) -> None: ...
    def Determine_Atom_Type(self, rule: str) -> None: ...
    def save_as_mol2(self, filename: str, atomtype: str = "sybyl") -> None: ...
    def Save_As_Mol2(self, filename: str, atomtype: str = "sybyl") -> None: ...
    def to_residuetype(
        self, name: str | None = None, charge: Sequence[float] | None = None
    ) -> ResidueType: ...


class GaffParameters: ...


def get_assignment_from_mol2(file: str, total_charge: int | None = None) -> Assign: ...
def Get_Assignment_From_Mol2(file: str, total_charge: int | None = None) -> Assign: ...
def generate_gaff_frcmod(
    ifname: str,
    ofname: str,
    *,
    ffset: int = 1,
    print_all: bool = False,
    print_dihedral_contain_X: bool = True,
    datapath: str | None = None,
) -> None: ...
def load_gaff_parameters(dat_path: str, frcmod_path: str | None = None) -> GaffParameters: ...
def load_frcmod(filename: str) -> _FrcmodSections: ...
def save_sponge_input(
    molecule: Any,
    output_dir: str = ".",
    parameters: GaffParameters | None = None,
    prefix: str = "xponge",
    box: Sequence[float] | None = None,
) -> None: ...
def save_amber_sponge_input(
    molecule: Any,
    data: Mapping[str, Any],
    output_dir: str = ".",
    prefix: str = "xponge",
    box: Sequence[float] | None = None,
    cmap_source: str = "",
) -> None: ...
def assignment_to_residue_type(
    assign: Assign, name: str | None = None, charge: Sequence[float] | None = None
) -> ResidueType: ...
