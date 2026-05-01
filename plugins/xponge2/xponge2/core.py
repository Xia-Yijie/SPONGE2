from dataclasses import dataclass, field


@dataclass
class Atom:
    name: str
    type: str
    x: float
    y: float
    z: float
    charge: float = 0.0
    mass: float = 0.0
    lj_type: str | None = None


@dataclass
class ResidueType:
    name: str = "MOL"
    atoms: list[Atom] = field(default_factory=list)
    bonds: list[tuple[int, int]] = field(default_factory=list)
    forcefield: str | None = None

    def add_atom(self, name, atom_type, x=0.0, y=0.0, z=0.0, charge=0.0, mass=0.0, lj_type=None):
        self.atoms.append(
            Atom(
                str(name),
                str(atom_type),
                float(x),
                float(y),
                float(z),
                float(charge),
                float(mass),
                str(lj_type) if lj_type is not None else str(atom_type),
            )
        )
        return self.atoms[-1]

    def add_connectivity(self, atom1, atom2):
        i = self._atom_index(atom1)
        j = self._atom_index(atom2)
        if i == j:
            raise ValueError("cannot connect an atom to itself")
        self.bonds.append((min(i, j), max(i, j)))

    def copy(self):
        residue = ResidueType(self.name, forcefield=self.forcefield)
        for attr in ("head", "tail"):
            if hasattr(self, attr):
                setattr(residue, attr, getattr(self, attr))
        for atom in self.atoms:
            residue.add_atom(
                atom.name,
                atom.type,
                atom.x,
                atom.y,
                atom.z,
                atom.charge,
                atom.mass,
                atom.lj_type,
            )
        residue.bonds = list(self.bonds)
        return residue

    def _atom_index(self, atom):
        if isinstance(atom, int):
            return atom
        return self.atoms.index(atom)

    def to_molecule(self):
        return Molecule(self)

    def __add__(self, other):
        return self.to_molecule() + other

    @classmethod
    def get_type(cls, name):
        try:
            return _RESIDUE_TYPE_REGISTRY[str(name)].copy()
        except KeyError as exc:
            raise KeyError(f"unknown residue type: {name}") from exc

    Get_Type = get_type


@dataclass
class Molecule:
    residues: list[ResidueType] = field(default_factory=list)

    def __init__(self, residue=None):
        self.residues = []
        if residue is not None:
            self.add_residue(residue)

    def add_residue(self, residue):
        if not isinstance(residue, ResidueType):
            raise TypeError("residue must be a ResidueType")
        self.residues.append(residue)

    @property
    def atoms(self):
        return [atom for residue in self.residues for atom in residue.atoms]

    @property
    def bonds(self):
        rows = []
        offset = 0
        for residue in self.residues:
            rows.extend((i + offset, j + offset) for i, j in residue.bonds)
            offset += len(residue.atoms)
        return rows

    def __add__(self, other):
        result = Molecule()
        result.residues.extend(self.residues)
        if isinstance(other, ResidueType):
            result.add_residue(other)
        elif isinstance(other, Molecule):
            result.residues.extend(other.residues)
        else:
            return NotImplemented
        return result

    @property
    def forcefield(self):
        forcefields = {residue.forcefield for residue in self.residues if residue.forcefield}
        if not forcefields:
            return None
        if len(forcefields) != 1:
            raise ValueError("molecule mixes residue types from multiple force fields")
        return next(iter(forcefields))

_RESIDUE_TYPE_REGISTRY = {}
