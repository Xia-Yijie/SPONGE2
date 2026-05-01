from ...core import ResidueType
from ...core import _RESIDUE_TYPE_REGISTRY


_AMBER_FORCEFIELD_REGISTRY = {}


PROTEIN_RESIDUES = [
    "ACE", "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY",
    "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL", "HIS", "NME",
]


def _tuple_keyed(table):
    return {
        tuple(key.split("-")) if isinstance(key, str) else tuple(key): value
        for key, value in table.items()
    }


def _normalise_amber_data(data):
    data = dict(data)
    for name in ("bond", "angle", "proper", "improper"):
        data[name] = _tuple_keyed(data[name])
    return data


def instantiate_amber_residue(forcefield, name, template):
    residue = ResidueType(name, forcefield=forcefield)
    residue.head = template.get("head")
    residue.tail = template.get("tail")
    for atom in template["atoms"]:
        residue.add_atom(
            atom["name"],
            atom["type"],
            atom["x"],
            atom["y"],
            atom["z"],
            atom["charge"],
            atom["mass"],
            atom["lj_type"],
        )
    for atom1, atom2 in template["bonds"]:
        residue.add_connectivity(atom1, atom2)
    _RESIDUE_TYPE_REGISTRY[name] = residue


def register_amber_forcefield_data(forcefield, data, residue_names):
    forcefield = str(forcefield)
    data = _normalise_amber_data(data)
    _AMBER_FORCEFIELD_REGISTRY[forcefield] = data
    for name in residue_names:
        instantiate_amber_residue(forcefield, name, data["residues"][name])
