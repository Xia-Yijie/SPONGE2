from collections.abc import Mapping, Sequence
from typing import Any

from ...core import ResidueType

PROTEIN_RESIDUES: list[str]
_AMBER_FORCEFIELD_REGISTRY: dict[str, dict[str, Any]]

def instantiate_amber_residue(
    forcefield: str, name: str, template: Mapping[str, Any]
) -> None: ...
def register_amber_forcefield_data(
    forcefield: str, data: Mapping[str, Any], residue_names: Sequence[str]
) -> None: ...
