from collections.abc import Sequence

from ._core import Assign
from .core import ResidueType


def assignment_to_residue_type(
    assign: Assign, name: str | None = None, charge: Sequence[float] | None = None
) -> ResidueType: ...
