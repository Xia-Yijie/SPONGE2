"""bsc1 DNA residue and parameter declarations."""

from ._bsc1_data import DATA
from .registry import register_amber_forcefield_data


register_amber_forcefield_data("bsc1", DATA, ["DA5", "DA", "DT", "DC", "DG", "DA3"])
