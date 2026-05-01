"""ff14SB residue and parameter declarations."""

from ._ff14sb_data import DATA
from .registry import PROTEIN_RESIDUES, register_amber_forcefield_data


register_amber_forcefield_data("ff14sb", DATA, PROTEIN_RESIDUES)
