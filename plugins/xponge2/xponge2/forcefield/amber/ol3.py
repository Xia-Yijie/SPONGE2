"""OL3 RNA residue and parameter declarations."""

from ._ol3_data import DATA
from .registry import register_amber_forcefield_data


register_amber_forcefield_data("ol3", DATA, ["A5", "A", "U", "C", "G", "U3"])
