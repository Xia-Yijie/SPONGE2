"""ff19SB residue and parameter declarations."""

from ._ff19sb_data import DATA
from .registry import PROTEIN_RESIDUES, register_amber_forcefield_data


register_amber_forcefield_data("ff19sb", DATA, PROTEIN_RESIDUES)
