from typing import Self


class Atom:
    name: str
    type: str
    x: float
    y: float
    z: float
    charge: float
    mass: float
    lj_type: str | None

    def __init__(
        self,
        name: str,
        type: str,
        x: float,
        y: float,
        z: float,
        charge: float = 0.0,
        mass: float = 0.0,
        lj_type: str | None = None,
    ) -> None: ...


class ResidueType:
    name: str
    atoms: list[Atom]
    bonds: list[tuple[int, int]]
    forcefield: str | None
    head: str
    tail: str

    def __init__(
        self,
        name: str = "MOL",
        atoms: list[Atom] | None = None,
        bonds: list[tuple[int, int]] | None = None,
        forcefield: str | None = None,
    ) -> None: ...
    def add_atom(
        self,
        name: str,
        atom_type: str,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        charge: float = 0.0,
        mass: float = 0.0,
        lj_type: str | None = None,
    ) -> Atom: ...
    def add_connectivity(self, atom1: int | Atom, atom2: int | Atom) -> None: ...
    def copy(self) -> Self: ...
    def to_molecule(self) -> Molecule: ...
    def __add__(self, other: ResidueType | Molecule) -> Molecule: ...
    @classmethod
    def get_type(cls, name: str) -> Self: ...
    @classmethod
    def Get_Type(cls, name: str) -> Self: ...


class Molecule:
    residues: list[ResidueType]

    def __init__(self, residue: ResidueType | None = None) -> None: ...
    def add_residue(self, residue: ResidueType) -> None: ...
    @property
    def atoms(self) -> list[Atom]: ...
    @property
    def bonds(self) -> list[tuple[int, int]]: ...
    def __add__(self, other: ResidueType | Molecule) -> Molecule: ...
    @property
    def forcefield(self) -> str | None: ...


_RESIDUE_TYPE_REGISTRY: dict[str, ResidueType]
