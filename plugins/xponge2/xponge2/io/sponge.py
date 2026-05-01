import importlib
from pathlib import Path

from ..forcefield.amber.registry import _AMBER_FORCEFIELD_REGISTRY


def _ff19sb_cmap_path():
    return (
        Path(__file__).resolve().parents[1]
        / "forcefield"
        / "amber"
        / "files"
        / "ff19sb_cmap.txt"
    )


def save_sponge_input(molecule, output_dir=".", parameters=None, prefix="xponge", box=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    core = importlib.import_module("xponge2._core")

    if parameters is not None:
        return core.save_sponge_input(
            molecule,
            str(output_dir),
            parameters=parameters,
            prefix=prefix,
            box=box,
        )

    forcefield = molecule.forcefield
    if forcefield in _AMBER_FORCEFIELD_REGISTRY:
        cmap_source = str(_ff19sb_cmap_path()) if forcefield == "ff19sb" else ""
        return core.save_amber_sponge_input(
            molecule,
            _AMBER_FORCEFIELD_REGISTRY[forcefield],
            str(output_dir),
            prefix=prefix,
            box=box,
            cmap_source=cmap_source,
        )

    residue_names = [residue.name for residue in molecule.residues]
    raise NotImplementedError(
        "xponge2 has no registered AMBER parameterization path for "
        f"forcefield={forcefield or '<unset>'}, residues={residue_names}"
    )
