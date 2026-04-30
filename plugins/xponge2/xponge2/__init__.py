"""Compatibility layer for the C++ Xponge implementation."""

__version__ = "0.0.1"

try:
    from ._core import Assign
    from ._core import Get_Assignment_From_Mol2
    from ._core import get_assignment_from_mol2
except ImportError:
    # The extension is unavailable while build backends import this module for
    # metadata. Normal runtime imports will resolve these names from _core.
    pass

__all__ = [
    "Assign",
    "get_assignment_from_mol2",
    "Get_Assignment_From_Mol2",
]
